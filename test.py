import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from agent_config import AGENT_INSTRUCTIONS


MODEL = "gpt-5-nano"
VALID_LABELS = {"yes", "no", "maybe"}

# 에이전트의 벡터 검색에서 실제로 반환됐던 문서와 순서를 고정한다.
FIXED_RETRIEVAL_RESULTS = (
    {"pmid": "16418930", "similarity": 0.6744},
    {"pmid": "26686513", "similarity": 0.4891},
    {"pmid": "27757987", "similarity": 0.4453},
)


COMPARISON_INSTRUCTIONS = """
제공된 근거만 사용하여 생의학 연구 질문에 답하세요.

다음 차이 비교 규칙을 일반적인 판정 규칙보다 먼저 적용하세요.

두 측정값, 집단, 치료법 또는 치료 결과의 차이를 묻는 질문에서는
단순히 두 수치가 다르다는 이유만으로 yes라고 답하지 마세요.

연구 결과가 차이를 small, slight, minor 또는 negligible로 표현하거나
두 결과가 similar, comparable 또는 broadly equal하다고 설명하면
no로 답하세요.

연구 결과가 의미 있는 차이, 통계적으로 유의한 차이 또는
임상적으로 유의한 차이를 명확하게 보고하면 yes로 답하세요.

수치 차이만 제시되어 있고 그 차이의 의미나 중요성을
판단할 수 없다면 maybe로 답하세요.

위의 차이 비교 규칙에 해당하지 않는 일반 질문에서는
연구 결과가 질문의 핵심 주장을 지지하면 yes로 답하고,
명확하게 부정하거나 반박하면 no로 답하세요.

직접적인 근거가 부족하거나 관련된 결과가 서로 충돌하면
maybe로 답하세요.

최종 답변에는 소문자 yes, no, maybe 중 하나만 출력하세요.
설명이나 다른 문장은 출력하지 마세요.
"""


def load_agent_instructions() -> str:
    """모델을 로딩하거나 검색을 실행하지 않고 현재 지침을 가져온다."""
    return AGENT_INSTRUCTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PubMedQA 2번 문항의 판정 프롬프트 비교",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="각 프롬프트의 반복 호출 횟수 (기본값: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 1 <= args.runs <= 20:
        raise SystemExit("--runs는 1에서 20 사이여야 합니다.")

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(".env 파일에 OPENAI_API_KEY를 입력하세요.")

    client = OpenAI()

    prompts = {
        "comparison_rule": COMPARISON_INSTRUCTIONS,
        "agent_full": load_agent_instructions(),
    }

    dataset = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_labeled",
        split="train",
    )

    # PubMedQA의 두 번째 문항만 고정해서 사용한다.
    sample = dataset[1]

    question = sample["question"]

    fixed_pmids = {
        item["pmid"]
        for item in FIXED_RETRIEVAL_RESULTS
    }
    documents_by_pmid: dict[str, str] = {}

    # 검색하지 않고 데이터셋에서 지정된 PMID 세 개의 context를 꺼낸다.
    for dataset_sample in dataset:
        pmid = str(dataset_sample["pubid"])

        if pmid in fixed_pmids:
            documents_by_pmid[pmid] = "\n\n".join(
                dataset_sample["context"]["contexts"]
            )

        if len(documents_by_pmid) == len(fixed_pmids):
            break

    missing_pmids = fixed_pmids - documents_by_pmid.keys()

    if missing_pmids:
        raise RuntimeError(
            "고정 문서를 찾지 못했습니다: "
            + ", ".join(sorted(missing_pmids))
        )

    context_parts = []

    for item in FIXED_RETRIEVAL_RESULTS:
        pmid = item["pmid"]
        similarity = item["similarity"]

        context_parts.append(
            f"PMID : {pmid}\n"
            f"Similarity : {similarity:.4f}\n"
            f"Context : {documents_by_pmid[pmid]}"
        )

    context = "\n\n".join(context_parts)

    # 정답은 평가에만 사용하며 모델 입력에는 포함하지 않는다.
    gold_label = sample["final_decision"].lower()

    model_input = (
        f"Question:\n{question}\n\n"
        f"Evidence:\n{context}"
    )

    results: list[dict[str, object]] = []

    print(f"질문: {question}")
    print(f"정답: {gold_label}")
    print(
        "고정된 PMID: "
        f"{[item['pmid'] for item in FIXED_RETRIEVAL_RESULTS]}"
    )
    print(f"프롬프트당 반복 횟수: {args.runs}")

    # 두 프롬프트를 같은 질문과 같은 근거로 번갈아 호출한다.
    for run_number in range(1, args.runs + 1):
        for prompt_name, instructions in prompts.items():
            started_at = time.perf_counter()

            try:
                response = client.responses.create(
                    model=MODEL,
                    instructions=instructions,
                    input=model_input,
                )

                latency_sec = time.perf_counter() - started_at
                prediction = response.output_text.strip().lower()
                is_valid = prediction in VALID_LABELS
                is_correct = is_valid and prediction == gold_label

                row = {
                    "run": run_number,
                    "prompt": prompt_name,
                    "question": question,
                    "gold_label": gold_label,
                    "prediction": prediction,
                    "is_valid": is_valid,
                    "is_correct": is_correct,
                    "status": "completed",
                    "latency_sec": round(latency_sec, 3),
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "error": "",
                }

            except OpenAIError as error:
                latency_sec = time.perf_counter() - started_at

                row = {
                    "run": run_number,
                    "prompt": prompt_name,
                    "question": question,
                    "gold_label": gold_label,
                    "prediction": "",
                    "is_valid": False,
                    "is_correct": False,
                    "status": "api_error",
                    "latency_sec": round(latency_sec, 3),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "error": str(error),
                }

            results.append(row)

            print(
                f"실행 {run_number} | {prompt_name} | "
                f"답변: {row['prediction'] or 'API 오류'} | "
                f"정답 여부: {row['is_correct']}"
            )

    results_df = pd.DataFrame(results)

    summary_rows: list[dict[str, object]] = []

    for prompt_name in prompts:
        prompt_df = results_df[
            results_df["prompt"] == prompt_name
        ]
        completed_df = prompt_df[
            prompt_df["status"] == "completed"
        ]

        summary_rows.append(
            {
                "prompt": prompt_name,
                "requested_runs": args.runs,
                "completed_runs": len(completed_df),
                "yes_count": int(
                    (completed_df["prediction"] == "yes").sum()
                ),
                "no_count": int(
                    (completed_df["prediction"] == "no").sum()
                ),
                "maybe_count": int(
                    (completed_df["prediction"] == "maybe").sum()
                ),
                "correct_count": int(
                    completed_df["is_correct"].sum()
                ),
                "accuracy": (
                    completed_df["is_correct"].mean()
                    if len(completed_df) > 0
                    else None
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    run_dir = (
        Path("results")
        / datetime.now().strftime("prompt_test_q2_%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    results_path = run_dir / "results.csv"
    summary_path = run_dir / "summary.csv"
    config_path = run_dir / "config.json"

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model": MODEL,
                "dataset_index": 1,
                "question_number": 2,
                "question": question,
                "evidence_mode": "fixed_vector_top3",
                "fixed_retrieval_results": FIXED_RETRIEVAL_RESULTS,
                "runs_per_prompt": args.runs,
                "prompts": prompts,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 60)
    print(summary_df.to_string(index=False))
    print(f"상세 결과: {results_path}")
    print(f"요약 결과: {summary_path}")
    print(f"실험 설정: {config_path}")


if __name__ == "__main__":
    main()
