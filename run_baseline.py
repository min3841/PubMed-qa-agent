import os  # 환경 변수 등 운영체제의 기능을 사용
import pandas as pd  # 평가 결과를 표 형태로 만들고 CSV 파일로 저장
import argparse  # 실행 명령어에서 --num-samples 같은 옵션을 입력받음
import time  # API 응답 시간을 측정

from openai import OpenAI  # OpenAI API를 호출하기 위한 클래스
from pathlib import Path  # 폴더 및 파일 경로를 만들고 관리
from dotenv import load_dotenv  # .env 파일에 저장된 환경 변수를 불러옴
from datasets import load_dataset  # Hugging Face 데이터셋을 불러옴
from tqdm import tqdm #작업이 얼마나 진행됐는지 진행률 표시줄


MODEL = "gpt-5-nano"
VALID_LABELS = {"yes", "no", "maybe"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="단일 LLM 평가"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="평가할 문항 수 (기본값: 5)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 1 <= args.num_samples <= 100:
        raise SystemExit(
            "--num-samples는 1에서 100 사이여야 합니다."
        )

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("API 키가 존재하지 않습니다.")

    client = OpenAI()

    dataset = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_labeled",
        split="train",
    ).select(
        range(args.num_samples)
    )

    print(f"평가할 데이터 수: {len(dataset)}")

    results: list[dict[str, object]] = []

    for index, sample in enumerate(
        tqdm(dataset, desc="단일 LLM 평가"),
        start=1,
    ):
        question = sample["question"]
        gold_label = sample["final_decision"].lower()

        started_at = time.perf_counter()

        response = client.responses.create(
            model=MODEL,
            instructions=(
                "Answer the biomedical research question. "
                "Return exactly one lowercase label: "
                "yes, no, or maybe. "
                "Do not provide an explanation."
            ),
            input=question,
        )

        latency_sec = (
            time.perf_counter() - started_at
        )

        prediction = (
            response.output_text.strip().lower()
        )

        is_valid = prediction in VALID_LABELS
        is_correct = (
            is_valid and prediction == gold_label
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = response.usage.total_tokens

        results.append(
            {
                "index": index,
                "pubid": sample["pubid"],
                "question": question,
                "gold_label": gold_label,
                "prediction": prediction,
                "is_valid": is_valid,
                "is_correct": is_correct,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_sec": round(
                    latency_sec,
                    3,
                ),
            }
        )

        print(f"\n문제: {index}")
        print(f"질문: {question}")
        print(f"정답: {gold_label}")
        print(f"모델 답: {prediction}")
        print(f"형식 준수 여부: {is_valid}")
        print(f"정답 여부: {is_correct}")
        print(f"응답 시간: {latency_sec:.3f}초")
        print(f"입력 토큰: {input_tokens}")
        print(f"출력 토큰: {output_tokens}")
        print(f"전체 토큰: {total_tokens}")

    results_df = pd.DataFrame(results)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_path = (
        results_dir
        / f"baseline_{args.num_samples}.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("단일 LLM 평가 완료")
    print(f"평가 문제 수: {len(results_df)}")
    print(
        f"정확도: "
        f"{results_df['is_correct'].mean():.1%}"
    )
    print(
        f"평균 응답 시간: "
        f"{results_df['latency_sec'].mean():.3f}초"
    )
    print(
        f"전체 입력 토큰: "
        f"{results_df['input_tokens'].sum()}"
    )
    print(
        f"전체 출력 토큰: "
        f"{results_df['output_tokens'].sum()}"
    )
    print(
        f"전체 사용 토큰: "
        f"{results_df['total_tokens'].sum()}"
    )
    print(f"결과 저장 위치: {output_path}")


if __name__ == "__main__":  #파일이 직접 실행이 되었을 때 main() 실행
    main()