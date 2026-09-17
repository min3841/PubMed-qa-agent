"""PubMedQA 도구 선택형 에이전트 실행기.

실행 순서:
1. 검색 담당 LLM이 vector_search 또는 pubmed_search를 선택한다.
2. 검색 문서를 CrossEncoder로 재정렬한다.
3. 가장 관련성이 높은 논문 1개만 최종 판정 LLM에 전달한다.
4. 문항별 결과와 전체 실행 과정을 CSV·JSONL로 저장한다.
"""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from requests import RequestException
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from agent_config import (
    EMBEDDING_MODEL,
    FINAL_ANSWER_INSTRUCTIONS,
    LABEL,
    MODEL,
    PUBMED_CANDIDATE_K,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
    SEARCH_INSTRUCTIONS,
    TOOLS,
)
from agent_search import append_log, prepare_search_resources, run_search_agent


BASE_DIR = Path(__file__).resolve().parent # 현재 폴더 위치


# ---------------------------------------------------------------------------
# 실행 옵션과 결과 저장
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="검색 후 CrossEncoder 상위 1개로 답하는 PubMedQA 에이전트",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="처음부터 불러올 문항 수 (기본값: 5, 최대: 100)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="평가를 시작할 문항 번호 (1부터 시작)", #특정 구간만 따로 실행하는 옵션
    )
    return parser.parse_args()


def create_run_paths(
    num_samples: int,
    start_index: int,
    evaluation_count: int,
) -> tuple[Path, Path]:
    """실행별 폴더를 만들고 사용한 설정을 trace.jsonl에 기록한다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        BASE_DIR
        / "results"
        / f"에이전트_리랭커상위1개_{num_samples}문항_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_path = run_dir / "trace.jsonl"
    output_path = run_dir / "results.csv"

    append_log(
        trace_path,
    {
        "event": "run_config",  # 실행 설정 기록
        "model": MODEL,  # 검색 및 최종 판정 LLM
        "embedding_model": EMBEDDING_MODEL,  # 질문과 문서를 벡터로 변환하는 모델
        "reranker_model": RERANKER_MODEL,  # 검색 문서를 재정렬하는 CrossEncoder
        "num_samples": num_samples,  # 불러올 마지막 문항 번호
        "start_index": start_index,  # 평가 시작 문항 번호
        "evaluation_count": evaluation_count,  # 실제 평가할 문항 수
        "architecture": "search_then_cross_encoder_top1_then_final_judge",   # 검색 → 리랭크 상위 1개 → 최종 판정 구조
        "pubmed_candidate_k": PUBMED_CANDIDATE_K,  # PubMed에서 가져올 후보 수
        "retrieval_top_k": RETRIEVAL_TOP_K,  # 벡터 검색으로 가져올 후보 수
        "reranker_top_k": 1,  # 리랭크 후 최종 판정에 전달할 문서 수
        "search_instructions": SEARCH_INSTRUCTIONS,  # 검색 담당 LLM 프롬프트
        "final_answer_instructions": FINAL_ANSWER_INSTRUCTIONS,  # 판정 LLM 프롬프트
        "tools": TOOLS,  # 검색 담당 LLM이 사용할 수 있는 도구
    }
    )
    print(f"실행 기록: {run_dir}")
    return trace_path, output_path


def create_result_row(index: int, sample) -> dict[str, object]:
    """정상 응답과 오류가 항상 같은 CSV 열을 갖도록 행을 초기화한다."""
    return {
        "index": index,
        "pubid": str(sample["pubid"]),
        "question": sample["question"],
        "gold_label": sample["final_decision"].lower(),
        "prediction": "",
        "raw_answer": "",
        "response_status": "",
        "is_valid": False,
        "is_correct": False,
        "used_tools": "",
        "search_count": 0,
        "search_trace": "",
        "error_type": "",
        "error_message": "",
        "reranker_model": RERANKER_MODEL,
        "reranker_candidate_pmids": "",
        "reranker_ranked_pmids": "",
        "reranker_scores": "",
        "reranker_selected_pmid": "",
        "reranker_selected_score": "",
        "reranker_selected_tool": "",
        "gold_pmid_in_reranker_candidates": False,
        "gold_pmid_selected": False,
    }


def save_result(
    result_row: dict[str, object],
    trace_path: Path,
    output_path: Path,
    response_id: str | None,
) -> None:
    """문항 하나가 끝날 때마다 JSONL과 CSV에 즉시 저장한다."""
    event = (
        "api_error"
        if result_row["response_status"] == "api_error"
        else "answer"
    )
    append_log(
        trace_path,
        {"event": event, "response_id": response_id, **result_row},
    )

    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_row))
        if write_header:
            writer.writeheader()
        writer.writerow(result_row)


def print_summary(results: list[dict[str, object]], output_path: Path) -> None:
    """전체 실행 결과를 간단히 출력한다."""
    total_count = len(results)
    api_error_count = sum(
        result["response_status"] == "api_error" for result in results
    )
    completed_count = sum(
        result["response_status"] == "completed" for result in results
    )
    correct_count = sum(bool(result["is_correct"]) for result in results)
    format_error_count = sum(
        result["response_status"] == "completed" and not result["is_valid"]
        for result in results
    )

    print("\n" + "=" * 60)
    print("에이전트 평가 종료")
    print(f"처리한 문항 수: {total_count}")
    print(f"답변 완료: {completed_count}")
    print(f"정답 수: {correct_count}")
    print(f"API 오류: {api_error_count}")
    print(f"답변 형식 오류: {format_error_count}")
    if total_count:
        print(f"전체 문항 기준 정확도: {correct_count / total_count:.1%}")
    if completed_count:
        print(f"답변 완료 문항 기준 정확도: {correct_count / completed_count:.1%}")
    print(f"결과 저장 위치: {output_path}")


# ---------------------------------------------------------------------------
# CrossEncoder 리랭크
# ---------------------------------------------------------------------------

def collect_documents(
    search_trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """모든 검색 문서를 모으고 중복 PMID는 한 번만 남긴다."""
    documents: list[dict[str, Any]] = []
    seen_pmids: set[str] = set()  #set: 중복값 저장 x

    for search_record in search_trace:
        for document in search_record.get("documents", []):
            if document["pmid"] in seen_pmids:
                continue
            seen_pmids.add(document["pmid"])
            documents.append({
                **document, # 기존 딕셔너리의 내용
                "search_index": search_record["search_index"],
                "tool": search_record["tool"],
            })

    if not documents:
        raise ValueError("리랭크할 검색 문서가 없습니다.")
    return documents


def rerank_top_one(
    question: str,
    search_trace: list[dict[str, Any]],
    reranker: CrossEncoder,
) -> dict[str, Any]:
    """질문과 각 문서를 함께 평가하고 가장 관련성 높은 문서 하나를 고른다."""
    documents = collect_documents(search_trace)
    pairs = [(question, document["chunk"]) for document in documents]
    scores = [float(score) for score in reranker.predict(pairs)]
    ranked = sorted(
        zip(documents, scores),
        key=lambda document_score: document_score[1],
        reverse=True,
    )
    selected_document, selected_score = ranked[0]

    return {
        "candidate_pmids": [document["pmid"] for document in documents],
        "ranked_pmids": [document["pmid"] for document, _ in ranked],
        "ranked_scores": [round(score, 6) for _, score in ranked],
        "selected_pmid": selected_document["pmid"],
        "selected_score": round(selected_score, 6),
        "selected_tool": selected_document["tool"],
        "collected_context": (
            f"Search {selected_document['search_index']}\n"
            f"Tool: {selected_document['tool']}\n"
            f"Retrieved PMIDs: {[selected_document['pmid']]}\n"
            f"Evidence:\n{selected_document['chunk']}"
        ),
    }


# ---------------------------------------------------------------------------
# 최종 판정과 문항 실행
# ---------------------------------------------------------------------------

def run_final_judge(
    client: OpenAI,
    question: str,
    collected_context: str,
) -> dict[str, object]:
    """검색 대화와 분리된 LLM이 선택된 논문만 읽고 최종 답을 고른다."""
    response = client.responses.create(
        model=MODEL,
        instructions=FINAL_ANSWER_INSTRUCTIONS,
        input=f"Question:\n{question}\n\nEvidence:\n{collected_context}",
    )
    prediction = response.output_text.strip().lower()
    return {
        "response_id": response.id,
        "prediction": prediction,
        "raw_answer": response.output_text,
        "response_status": response.status,
        "is_valid": response.status == "completed" and prediction in LABEL,
    }


def evaluate_sample( # 함수를 호출할 때 각 값의 이름 명시
    *,
    index: int,
    sample,
    client: OpenAI,
    search_resources: dict,
    reranker: CrossEncoder,
    trace_path: Path,
    output_path: Path,
) -> dict[str, object]:

    """한 문항의 검색, 리랭크, 판정, 저장을 순서대로 수행한다."""
    result_row = create_result_row(index, sample)
    question = str(result_row["question"])
    gold_label = str(result_row["gold_label"])
    search_trace: list[dict[str, Any]] = []
    response_id: str | None = None

    try:
        search_result = run_search_agent(
            client=client,
            question=question,
            search_resources=search_resources,
            index=index,
            pubid=str(result_row["pubid"]),
            trace_path=trace_path,
            search_trace=search_trace,
        )

        rerank_result = rerank_top_one(
            question=question,
            search_trace=search_result["search_trace"],
            reranker=reranker,
        )
        candidate_pmids = rerank_result["candidate_pmids"]
        selected_pmid = rerank_result["selected_pmid"]
        result_row.update(
            {
                "reranker_candidate_pmids": json.dumps(
                    candidate_pmids, ensure_ascii=False,
                ),
                "reranker_ranked_pmids": json.dumps(
                    rerank_result["ranked_pmids"], ensure_ascii=False,
                ),
                "reranker_scores": json.dumps(
                    rerank_result["ranked_scores"], ensure_ascii=False,
                ),
                "reranker_selected_pmid": selected_pmid,
                "reranker_selected_score": rerank_result["selected_score"],
                "reranker_selected_tool": rerank_result["selected_tool"],
                "gold_pmid_in_reranker_candidates": (
                    str(result_row["pubid"]) in candidate_pmids
                ),
                "gold_pmid_selected": str(result_row["pubid"]) == selected_pmid,
            }
        )

        append_log(
            trace_path,
            {
                "event": "rerank",
                "index": index,
                "pubid": result_row["pubid"],
                "question": question,
                **rerank_result,
            },
        )

        answer_result = run_final_judge(
            client=client,
            question=question,
            collected_context=str(rerank_result["collected_context"]),
        )
        response_id = str(answer_result.pop("response_id"))
        result_row.update(answer_result)
        result_row["is_correct"] = (
            bool(result_row["is_valid"])
            and result_row["prediction"] == gold_label
        )

        print(f"\n사용한 도구 순서: {search_result['used_tools']}")
        print(f"리랭커 후보 PMID: {candidate_pmids}")
        print(f"리랭커 순위: {rerank_result['ranked_pmids']}")
        print(f"리랭커 선택 PMID: {selected_pmid}")
        print(f"정답 PMID 선택 여부: {result_row['gold_pmid_selected']}")
        print(f"정답: {gold_label}")
        print(f"모델 답변: {result_row['prediction']}")
        print(f"형식 준수 여부: {result_row['is_valid']}")
        print(f"정답 여부: {result_row['is_correct']}")

    except (OpenAIError, RequestException) as error:
        result_row.update(
            {
                "response_status": "api_error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        print(f"\n문제 {index}: API 오류 발생")
        print(f"오류 종류: {type(error).__name__}")
        print("오류를 저장하고 다음 문항으로 넘어갑니다.")

    result_row.update(
        {
            "used_tools": ",".join(record["tool"] for record in search_trace),
            "search_count": len(search_trace),
            "search_trace": json.dumps(search_trace, ensure_ascii=False),
        }
    )
    save_result(result_row, trace_path, output_path, response_id)
    return result_row


# ---------------------------------------------------------------------------
# 프로그램 시작점
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if not 1 <= args.num_samples <= 100:
        raise SystemExit("num-samples는 1부터 100 사이여야 합니다.")
    if not 1 <= args.start_index <= args.num_samples:
        raise SystemExit("start-index는 1부터 num-samples 사이여야 합니다.")

    load_dotenv(BASE_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(".env 파일에 OPENAI_API_KEY를 입력하세요.")
    if not os.getenv("NCBI_EMAIL"):
        raise SystemExit(".env 파일에 NCBI_EMAIL을 입력하세요.")

    client = OpenAI()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    search_resources = prepare_search_resources(dataset, embedding_model)
    evaluation_data = dataset.select(
        range(args.start_index - 1, args.num_samples)
    )
    trace_path, output_path = create_run_paths(
        args.num_samples,
        args.start_index,
        len(evaluation_data),
    )

    results: list[dict[str, object]] = []
    for index, sample in enumerate(
        tqdm(evaluation_data, desc="리랭커 에이전트 평가"),
        start=args.start_index,
    ):
        results.append(
            evaluate_sample(
                index=index,
                sample=sample,
                client=client,
                search_resources=search_resources, #dataset, embedding_model
                reranker=reranker,
                trace_path=trace_path,
                output_path=output_path,
            )
        )

    print_summary(results, output_path)


if __name__ == "__main__":
    main()
