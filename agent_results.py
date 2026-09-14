"""실행 폴더 생성, JSONL·CSV 저장과 평가 요약 출력."""
import csv
import json
from datetime import datetime
from pathlib import Path

from agent_config import (
    MODEL, PUBMED_CANDIDATE_K, RETRIEVAL_TOP_K,
    SEARCH_INSTRUCTIONS, FINAL_ANSWER_INSTRUCTIONS, TOOLS,
)


def append_log(path: Path, record: dict) -> None:
    # 검색 직후 저장하여 다음 API 호출이 실패해도 근거를 보존한다.
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")



def create_run_paths(base_dir: Path, num_samples: int, start_index: int,
                     evaluation_count: int) -> tuple[Path, Path]:
    """실행별 설정과 저장 경로를 만든다."""
    # 실행별 폴더를 만들어 이전 결과를 덮어쓰지 않는다.
    run_dir = base_dir / "results" / datetime.now().strftime("agent_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_path = run_dir / "trace.jsonl"
    output_path = run_dir / "results.csv"
    append_log(trace_path, {
        "event": "run_config",
        "model": MODEL,
        "num_samples": num_samples,
        "start_index": start_index,
        "evaluation_count": evaluation_count,
        "variant": "pubmed_cosine_rerank_v1",
        "architecture": "separate_search_and_final_judge_v2_vector_nonempty_stop",
        "pubmed_candidate_k": PUBMED_CANDIDATE_K,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "search_instructions": SEARCH_INSTRUCTIONS,
        "final_answer_instructions": FINAL_ANSWER_INSTRUCTIONS,
        "tools": TOOLS,
    })
    print(f"실행 기록: {run_dir}")

    return trace_path, output_path


def create_result_row(index: int, sample) -> dict:
    """정상 응답과 API 오류가 같은 열 순서로 저장되도록 초기화한다."""
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
    }


def save_result(result_row: dict, trace_path: Path, output_path: Path,
                response_id: str | None = None) -> None:
    """문항 하나가 끝날 때마다 결과를 저장한다. 오류 문항도 포함한다."""
    event = (
        "api_error"
        if result_row["response_status"] == "api_error"
        else "answer"
    )

    append_log(
        trace_path,
        {
            "event": event,
            "response_id": response_id,
            **result_row,
        },
    )

    write_header = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )

    with output_path.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(result_row),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(result_row)



def print_summary(results: list[dict], output_path: Path) -> None:
    total_count = len(results)

    api_error_count = sum(
        result["response_status"] == "api_error"
        for result in results
    )

    completed_count = sum(
        result["response_status"] == "completed"
        for result in results
    )

    correct_count = sum(
        result["is_correct"]
        for result in results
    )

    format_error_count = sum(
        result["response_status"] == "completed"
        and not result["is_valid"]
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
        print(
            "전체 문항 기준 정답 비율: "
            f"{correct_count / total_count:.1%}"
        )

    if completed_count:
        print(
            "답변 완료 문항 기준 정확도: "
            f"{correct_count / completed_count:.1%}"
        )

    print(f"결과 저장 위치: {output_path}")
