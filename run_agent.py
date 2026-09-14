"""PubMedQA 에이전트 평가: 준비 → 검색 → 최종 판정 → 결과 저장."""
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from requests import RequestException
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 설정과 프롬프트는 agent_config.py에서 관리한다.
# 기존에 run_agent에서 지침을 import하던 코드도 계속 사용할 수 있다.
from agent_config import (
    AGENT_INSTRUCTIONS, FINAL_ANSWER_INSTRUCTIONS, SEARCH_INSTRUCTIONS,
)
from agent_judge import run_final_judge
from agent_results import (
    create_result_row, create_run_paths, print_summary, save_result,
)
from agent_search import prepare_search_resources, run_search_agent

BASE_DIR = Path(__file__).resolve().parent


def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="검색 도구 선택 에이전트 평가")
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="평가 문항 수 (기본값: 5)",
    )
    parser.add_argument(
        "--start-index", type=int, default=1,
        help="평가를 시작할 문항 번호 (1부터 시작, num-samples까지 실행)",
    )
    return parser.parse_args()


def main() -> None:
    arg = parser()
    start_index = getattr(arg, "start_index", 1)
    if not 1 <= start_index <= arg.num_samples:
        raise SystemExit("start-index는 1부터 num-samples 사이여야 합니다.")
    if not 1 <= arg.num_samples <= 100:
        raise SystemExit("num_samples는 1부터 100 사이여야 합니다.")

    load_dotenv(BASE_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(".env 파일에 OPENAI_API_KEY를 입력하세요.")
    if not os.getenv("NCBI_EMAIL"):
        raise SystemExit(".env 파일에 NCBI_EMAIL을 입력하세요.")

    client = OpenAI()
    embedding_model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
    datasets = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    # 검색 문서는 전체 데이터로 준비하고, 평가 질문만 지정한 범위로 선택한다.
    search_resources = prepare_search_resources(datasets, embedding_model)
    num_dataset = datasets.select(range(start_index - 1, arg.num_samples))
    trace_path, output_path = create_run_paths(
        BASE_DIR, arg.num_samples, start_index, len(num_dataset),
    )
    results: list[dict[str, object]] = []

    for index, sample in enumerate(
        tqdm(num_dataset, desc="에이전트 평가"), start=start_index,
    ):
        result_row = create_result_row(index, sample)
        question = result_row["question"]
        gold_label = result_row["gold_label"]
        search_trace = []
        response_id = None

        try:
            # 검색 기록 리스트를 공유해 중간 오류가 나도 받은 근거를 보존한다.
            search_result = run_search_agent(
                client=client,
                question=question,
                search_resources=search_resources,
                index=index,
                pubid=result_row["pubid"],
                trace_path=trace_path,
                search_trace=search_trace,
            )
            answer_result = run_final_judge(
                client=client,
                question=question,
                collected_context=search_result["collected_context"],
            )
            response_id = answer_result.pop("response_id")
            result_row.update(answer_result)
            result_row["is_correct"] = (
                result_row["is_valid"] and result_row["prediction"] == gold_label
            )

            print(f"\n사용한 도구 순서: {search_result['used_tools']}")
            print(f"정답: {gold_label}")
            print(f"모델 답변: {result_row['prediction']}")
            print(f"형식 준수 여부: {result_row['is_valid']}")
            print(f"정답 여부: {result_row['is_correct']}")

        except (OpenAIError, RequestException) as error:
            # 요청 오류만 기록하고 다음 문항으로 진행한다.
            result_row.update({
                "response_status": "api_error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
            print(f"\n문제 {index}: API 오류 발생")
            print(f"오류 종류: {type(error).__name__}")
            print("오류를 저장하고 다음 문항으로 넘어갑니다.")

        # 성공·오류 여부와 관계없이 완료된 검색과 현재 문항 결과를 저장한다.
        result_row.update({
            "used_tools": ",".join(record["tool"] for record in search_trace),
            "search_count": len(search_trace),
            "search_trace": json.dumps(search_trace, ensure_ascii=False),
        })
        results.append(result_row)
        save_result(result_row, trace_path, output_path, response_id)

    print_summary(results, output_path)


if __name__ == "__main__":
    main()
