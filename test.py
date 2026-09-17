"""마지막 에이전트 오답을 고정 검색 문서로 다시 판정하는 비교 실험.

1. 정답 PMID 문서만 제공
2. 마지막 실행의 검색 문서 전체를 같은 순서로 제공
3. 같은 문서를 CrossEncoder로 재정렬하고 상위 1개만 제공

세 조건 모두 마지막 실행의 최종 판정 모델과 지침을 그대로 사용한다.
정답 라벨은 채점에만 사용하며 모델 입력에는 포함하지 않는다.
"""

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from sentence_transformers import CrossEncoder

from agent_config import FINAL_ANSWER_INSTRUCTIONS, LABEL, MODEL


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_TRACE = (
    BASE_DIR
    / "results"
    / "최종에이전트_100문항_정확도68퍼"
    / "trace.jsonl"
)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

CONDITIONS = {
    "1": {
        "name": "gold_only",
        "folder": "1_정답PMID문서만",
        "description": "정답 PMID 문서 하나만 제공",
    },
    "2": {
        "name": "fixed_all",
        "folder": "2_기존검색문서전체",
        "description": "마지막 실행의 검색 문서 전체와 순서를 그대로 제공",
    },
    "3": {
        "name": "reranker_top1",
        "folder": "3_리랭커상위1개문서",
        "description": "고정 문서를 CrossEncoder로 재정렬한 뒤 1위 문서만 제공",
    },
    "4": {
        "name": "reranker_top3",
        "folder": "4_리랭커상위3개문서",
        "description": "고정 문서를 CrossEncoder로 재정렬한 뒤 상위 3개 문서 제공",
    },
}

RESULT_FIELDS = [
    "index", "pubid", "question", "gold_label", "original_prediction",
    "original_used_tools", "condition", "condition_description",
    "original_pmids", "provided_pmids", "gold_pmid_provided",
    "reranker_ranked_pmids", "reranker_scores", "prediction", "raw_answer",
    "response_status", "is_valid", "is_correct", "latency_sec",
    "input_tokens", "output_tokens", "total_tokens", "error_type",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="정답 PMID가 검색된 에이전트 오답의 3조건 비교 실험",
    )
    parser.add_argument(
        "--source-trace", type=Path, default=DEFAULT_SOURCE_TRACE,
        help="마지막 100문항 실행의 trace.jsonl 경로",
    )
    parser.add_argument(
        "--condition",
        choices=("1", "2", "3", "4", "reranker", "all"),
        default="all",
        help=(
            "실행 조건. reranker는 리랭커 상위 1개와 3개를 비교하고, "
            "all은 모든 조건을 차례로 실행"
        ),
    )
    parser.add_argument(
        "--max-items", type=int, default=None,
        help="앞에서부터 실행할 최대 문항 수. 생략하면 대상 전체 실행",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="API를 호출하지 않고 대상 문항과 고정 문서만 점검",
    )
    return parser.parse_args()


def read_trace(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"원본 trace를 찾지 못했습니다: {path}")

    run_config: dict[str, Any] | None = None
    answers: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"trace {line_number}번째 줄의 JSON이 잘못됐습니다."
                ) from error

            if record.get("event") == "run_config" and run_config is None:
                run_config = record
            elif record.get("event") == "answer":
                answers.append(record)

    if run_config is None:
        raise ValueError("trace에서 run_config를 찾지 못했습니다.")
    if not answers:
        raise ValueError("trace에서 answer 기록을 찾지 못했습니다.")
    return run_config, answers


def parse_search_trace(answer: dict[str, Any]) -> list[dict[str, Any]]:
    value = answer.get("search_trace", [])
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise TypeError(
            f"{answer.get('index')}번 문항의 search_trace가 리스트가 아닙니다."
        )
    return value


def split_context_documents(
    context: str,
    search_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """벡터 및 PubMed 검색의 결합 문자열을 PMID별 원문 조각으로 나눈다."""
    starts = list(re.finditer(r"(?m)^PMID\s*:\s*(\d+)\s*$", context))
    documents: list[dict[str, Any]] = []

    for position, match in enumerate(starts):
        end = starts[position + 1].start() if position + 1 < len(starts) else len(context)
        chunk = context[match.start():end].strip()
        score_match = re.search(
            r"(?m)^Similarity\s*:\s*([^\n]+)$", chunk,
        )
        documents.append(
            {
                "pmid": match.group(1),
                "chunk": chunk,
                "reranker_text": chunk,
                "original_similarity": (
                    score_match.group(1).strip() if score_match else ""
                ),
                "search_index": search_record.get("search_index"),
                "tool": search_record.get("tool", ""),
                "query": search_record.get("query", ""),
            }
        )

    expected_pmids = [str(pmid) for pmid in search_record.get("pmids", [])]
    parsed_pmids = [document["pmid"] for document in documents]
    if expected_pmids and parsed_pmids != expected_pmids:
        raise ValueError(
            "검색 문서 분리 결과와 저장된 PMID 순서가 다릅니다. "
            f"expected={expected_pmids}, parsed={parsed_pmids}"
        )
    return documents


def collect_documents(
    search_trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen_pmids: set[str] = set()

    for search_record in search_trace:
        parsed = split_context_documents(
            search_record.get("context", ""), search_record,
        )
        for document in parsed:
            if document["pmid"] in seen_pmids:
                continue
            seen_pmids.add(document["pmid"])
            documents.append(document)
    return documents


def select_experiment_items(
    answers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    for answer in answers:
        if answer.get("is_correct") is not False:
            continue
        search_trace = parse_search_trace(answer)
        documents = collect_documents(search_trace)
        gold_pmid = str(answer["pubid"])
        if gold_pmid not in {document["pmid"] for document in documents}:
            continue
        selected.append(
            {"answer": answer, "search_trace": search_trace, "documents": documents}
        )
    return selected


def format_search_evidence(documents: list[dict[str, Any]]) -> str:
    """선택 문서를 원래 에이전트가 사용한 검색 근거 형식으로 만든다."""
    grouped: dict[tuple[Any, str, str], list[dict[str, Any]]] = defaultdict(list)
    group_order: list[tuple[Any, str, str]] = []

    for document in documents:
        key = (
            document.get("search_index"),
            document.get("tool", ""),
            document.get("query", ""),
        )
        if key not in grouped:
            group_order.append(key)
        grouped[key].append(document)

    evidence_parts: list[str] = []
    for search_index, tool, query in group_order:
        group = grouped[(search_index, tool, query)]
        pmids = [document["pmid"] for document in group]
        context = "\n\n".join(document["chunk"] for document in group)
        evidence_parts.append(
            f"Search {search_index}\n"
            f"Tool: {tool}\n"
            f"Retrieved PMIDs: {pmids}\n"
            f"Evidence:\n{context}"
        )
    return "\n\n".join(evidence_parts)


def format_original_evidence(search_trace: list[dict[str, Any]]) -> str:
    """마지막 실행 당시 최종 판정 LLM이 받은 근거를 그대로 재구성한다."""
    evidence_parts = []
    for record in search_trace:
        evidence_parts.append(
            f"Search {record['search_index']}\n"
            f"Tool: {record['tool']}\n"
            f"Retrieved PMIDs: {record['pmids']}\n"
            f"Evidence:\n{record['context']}"
        )
    return "\n\n".join(evidence_parts)


def choose_evidence(
    condition: str,
    item: dict[str, Any],
    reranker: CrossEncoder | None,
) -> tuple[str, list[str], list[str], list[float]]:
    answer = item["answer"]
    documents = item["documents"]
    gold_pmid = str(answer["pubid"])

    if condition == "1":
        selected = [
            document for document in documents if document["pmid"] == gold_pmid
        ][:1]
        return (
            format_search_evidence(selected),
            [document["pmid"] for document in selected], [], [],
        )

    if condition == "2":
        return (
            format_original_evidence(item["search_trace"]),
            [document["pmid"] for document in documents], [], [],
        )

    if reranker is None:
        raise RuntimeError("3번 조건에는 리랭커가 필요합니다.")

    pairs = [
        (answer["question"], document["reranker_text"])
        for document in documents
    ]
    scores = [float(score) for score in reranker.predict(pairs)]
    ranked = sorted(
        zip(documents, scores), key=lambda pair: pair[1], reverse=True,
    )
    top_k = 1 if condition == "3" else 3
    selected_documents = [document for document, _score in ranked[:top_k]]
    return (
        format_search_evidence(selected_documents),
        [document["pmid"] for document in selected_documents],
        [document["pmid"] for document, _score in ranked],
        [round(score, 6) for _document, score in ranked],
    )


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def result_row_base(
    item: dict[str, Any],
    condition: str,
    provided_pmids: list[str],
    ranked_pmids: list[str],
    reranker_scores: list[float],
) -> dict[str, Any]:
    answer = item["answer"]
    original_pmids = [document["pmid"] for document in item["documents"]]
    return {
        "index": answer["index"],
        "pubid": str(answer["pubid"]),
        "question": answer["question"],
        "gold_label": answer["gold_label"],
        "original_prediction": answer.get("prediction", ""),
        "original_used_tools": answer.get("used_tools", ""),
        "condition": CONDITIONS[condition]["name"],
        "condition_description": CONDITIONS[condition]["description"],
        "original_pmids": json.dumps(original_pmids, ensure_ascii=False),
        "provided_pmids": json.dumps(provided_pmids, ensure_ascii=False),
        "gold_pmid_provided": str(answer["pubid"]) in provided_pmids,
        "reranker_ranked_pmids": json.dumps(ranked_pmids, ensure_ascii=False),
        "reranker_scores": json.dumps(reranker_scores, ensure_ascii=False),
    }


def execute_condition(
    condition: str,
    items: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    instructions: str,
    run_dir: Path,
    reranker: CrossEncoder | None,
) -> list[dict[str, Any]]:
    condition_dir = run_dir / CONDITIONS[condition]["folder"]
    result_path = condition_dir / "results.csv"
    evidence_path = condition_dir / "evidence.jsonl"
    rows: list[dict[str, Any]] = []

    print("\n" + "=" * 72)
    print(f"조건 {condition}: {CONDITIONS[condition]['description']}")

    for order, item in enumerate(items, start=1):
        answer = item["answer"]
        evidence, provided_pmids, ranked_pmids, scores = choose_evidence(
            condition, item, reranker,
        )
        row = result_row_base(
            item, condition, provided_pmids, ranked_pmids, scores,
        )

        # API 오류가 나도 어떤 고정 근거가 사용됐는지 먼저 남긴다.
        append_jsonl(
            evidence_path,
            {
                "index": answer["index"],
                "pubid": str(answer["pubid"]),
                "condition": CONDITIONS[condition]["name"],
                "provided_pmids": provided_pmids,
                "reranker_ranked_pmids": ranked_pmids,
                "reranker_scores": scores,
                "evidence": evidence,
            },
        )

        started_at = time.perf_counter()
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=(
                    f"Question:\n{answer['question']}\n\n"
                    f"Evidence:\n{evidence}"
                ),
            )
            latency_sec = time.perf_counter() - started_at
            prediction = response.output_text.strip().lower()
            status = response.status
            is_valid = status == "completed" and prediction in LABEL
            row.update(
                {
                    "prediction": prediction,
                    "raw_answer": response.output_text,
                    "response_status": status,
                    "is_valid": is_valid,
                    "is_correct": is_valid and prediction == answer["gold_label"],
                    "latency_sec": round(latency_sec, 3),
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "error_type": "", "error_message": "",
                }
            )
        except OpenAIError as error:
            latency_sec = time.perf_counter() - started_at
            row.update(
                {
                    "prediction": "", "raw_answer": "",
                    "response_status": "api_error", "is_valid": False,
                    "is_correct": False, "latency_sec": round(latency_sec, 3),
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )

        rows.append(row)
        save_rows(result_path, rows)
        print(
            f"[{order}/{len(items)}] {answer['index']}번 | "
            f"제공 PMID {provided_pmids} | 정답 {answer['gold_label']} | "
            f"예측 {row['prediction'] or 'API 오류'} | 정답 여부 {row['is_correct']}"
        )
    return rows


def build_summary(
    condition_rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for condition, rows in condition_rows.items():
        completed = [row for row in rows if row["response_status"] == "completed"]
        valid = [row for row in completed if row["is_valid"]]
        correct = [row for row in valid if row["is_correct"]]
        gold_top1 = [row for row in rows if row["gold_pmid_provided"]]
        summary.append(
            {
                "condition": CONDITIONS[condition]["name"],
                "description": CONDITIONS[condition]["description"],
                "target_count": len(rows),
                "completed_count": len(completed),
                "valid_count": len(valid),
                "correct_count": len(correct),
                "accuracy": round(len(correct) / len(valid), 4) if valid else None,
                "gold_pmid_provided_count": len(gold_top1),
                "gold_pmid_provided_rate": (
                    round(len(gold_top1) / len(rows), 4) if rows else None
                ),
                "api_error_count": len(rows) - len(completed),
                "format_error_count": len(completed) - len(valid),
            }
        )
    return summary


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.max_items is not None and args.max_items < 1:
        raise SystemExit("--max-items는 1 이상이어야 합니다.")

    source_trace = args.source_trace.expanduser().resolve()
    run_config, answers = read_trace(source_trace)
    items = select_experiment_items(answers)
    if args.max_items is not None:
        items = items[:args.max_items]
    if not items:
        raise SystemExit(
            "틀렸고 정답 PMID가 검색 결과에 포함된 문항을 찾지 못했습니다."
        )

    source_model = str(run_config.get("model") or MODEL)
    source_instructions = str(
        run_config.get("final_answer_instructions") or FINAL_ANSWER_INSTRUCTIONS
    )
    if args.condition == "all":
        selected_conditions = ["1", "2", "3", "4"]
    elif args.condition == "reranker":
        selected_conditions = ["3", "4"]
    else:
        selected_conditions = [args.condition]

    print(f"원본 trace: {source_trace}")
    print(f"원본 오답 수: {sum(a.get('is_correct') is False for a in answers)}")
    print(f"정답 PMID가 포함된 오답 수: {len(items)}")
    print(f"실행 조건: {selected_conditions}")
    print(f"판정 모델: {source_model}")
    print(
        "현재 agent_config.py와 원본 실행 지침 일치: "
        f"{source_instructions == FINAL_ANSWER_INSTRUCTIONS}"
    )

    if args.dry_run:
        for item in items:
            answer = item["answer"]
            pmids = [document["pmid"] for document in item["documents"]]
            print(
                f"{answer['index']}번 | 정답 {answer['gold_label']} | "
                f"기존 예측 {answer['prediction']} | 고정 PMID {pmids}"
            )
        return

    load_dotenv(BASE_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(".env 파일에 OPENAI_API_KEY를 입력하세요.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = (
        "오답(정답PMID제공)결과_리랭커상위1개3개비교"
        if selected_conditions == ["3", "4"]
        else "오답(정답PMID제공)결과_전체조건비교"
    )
    run_dir = BASE_DIR / "results" / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source_trace": str(source_trace),
                "source_model": source_model,
                "source_final_answer_instructions": source_instructions,
                "current_instructions_match_source": (
                    source_instructions == FINAL_ANSWER_INSTRUCTIONS
                ),
                "selected_conditions": selected_conditions,
                "selected_question_indices": [
                    item["answer"]["index"] for item in items
                ],
                "selection_rule": (
                    "원본 실행 오답 중 정답 PMID가 검색 결과에 포함된 문항"
                ),
                "reranker_model": (
                    RERANKER_MODEL
                    if any(condition in selected_conditions for condition in ("3", "4"))
                    else None
                ),
                "reranker_top_k": (
                    [1, 3]
                    if selected_conditions == ["3", "4"]
                    else [
                        1 if condition == "3" else 3
                        for condition in selected_conditions
                        if condition in ("3", "4")
                    ]
                ),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    client = OpenAI()
    reranker = None
    if any(condition in selected_conditions for condition in ("3", "4")):
        print(f"리랭커 로딩: {RERANKER_MODEL}")
        reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

    condition_rows: dict[str, list[dict[str, Any]]] = {}
    for condition in selected_conditions:
        condition_rows[condition] = execute_condition(
            condition=condition,
            items=items,
            client=client,
            model=source_model,
            instructions=source_instructions,
            run_dir=run_dir,
            reranker=reranker,
        )

    summary = build_summary(condition_rows)
    write_summary(run_dir / "summary.csv", summary)

    print("\n" + "=" * 72)
    for row in summary:
        accuracy = (
            f"{row['accuracy'] * 100:.1f}%"
            if row["accuracy"] is not None else "계산 불가"
        )
        print(
            f"{row['condition']}: {row['correct_count']}/"
            f"{row['valid_count']} ({accuracy}), 정답 PMID 제공 "
            f"{row['gold_pmid_provided_count']}/{row['target_count']}"
        )
    print(f"결과 폴더: {run_dir}")


if __name__ == "__main__":
    main()
