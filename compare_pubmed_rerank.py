import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from tqdm import tqdm

from PM_RAG import fetch_pubmed_documents, search_pubmed


BASE_DIR = Path(__file__).resolve().parent
MODEL = "gpt-5-nano"
VALID_LABELS = {"yes", "no", "maybe"}
CANDIDATE_K = 10
TOP_K = 3
INSTRUCTIONS = (
    "Answer the biomedical question using only the provided PubMed evidence. "
    "If the evidence is insufficient, answer maybe. "
    "Return exactly one lowercase label: yes, no, or maybe. "
    "Do not provide an explanation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PubMed 기본 순위와 코사인 유사도 재정렬 비교"
    )
    parser.add_argument("--num-samples", type=int, default=10)
    return parser.parse_args()


def join_context(documents: list[dict[str, str]]) -> str:
    if not documents:
        return "No PubMed evidence was retrieved."
    return "\n\n".join(
        f"PMID: {document['pmid']}\n"
        f"Title: {document['title']}\n"
        f"Abstract: {document['abstract'] or 'No abstract available.'}"
        for document in documents
    )


def rerank_documents(
    question: str,
    documents: list[dict[str, str]],
    embedding_model,
) -> tuple[list[dict[str, str]], list[float]]:
    if not documents:
        return [], []

    texts = [
        f"{document['title']}\n\n{document['abstract']}".strip()
        for document in documents
    ]
    question_embedding = embedding_model.encode(
        [question], convert_to_tensor=True, show_progress_bar=False,
    )
    document_embeddings = embedding_model.encode(
        texts, convert_to_tensor=True, show_progress_bar=False,
    )
    hits = semantic_search(
        question_embedding,
        document_embeddings,
        top_k=len(documents),
    )[0]
    ranked = [documents[hit["corpus_id"]] for hit in hits]
    scores = [float(hit["score"]) for hit in hits]
    return ranked, scores


def ask_llm(client: OpenAI, question: str, documents: list[dict[str, str]]) -> dict:
    started_at = time.perf_counter()
    response = client.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=(
            f"Question:\n{question}\n\n"
            f"PubMed evidence:\n{join_context(documents)}"
        ),
    )
    latency_sec = time.perf_counter() - started_at
    prediction = response.output_text.strip().lower()
    return {
        "prediction": prediction,
        "is_valid": prediction in VALID_LABELS,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
        "latency_sec": round(latency_sec, 3),
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.num_samples <= 100:
        raise SystemExit("num-samples는 1부터 100 사이여야 합니다.")

    load_dotenv(BASE_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(".env 파일에 OPENAI_API_KEY를 입력하세요.")
    if not os.getenv("NCBI_EMAIL"):
        raise SystemExit(".env 파일에 NCBI_EMAIL을 입력하세요.")

    client = OpenAI()
    embedding_model = SentenceTransformer(
        "NeuML/pubmedbert-base-embeddings"
    )
    dataset = load_dataset(
        "qiaojin/PubMedQA", "pqa_labeled", split="train",
    ).select(range(args.num_samples))

    run_dir = BASE_DIR / "results" / datetime.now().strftime(
        "pubmed_rerank_compare_%Y%m%d_%H%M%S_%f"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    output_path = run_dir / "results.csv"
    trace_path = run_dir / "retrieval.jsonl"
    rows: list[dict] = []

    for index, sample in enumerate(
        tqdm(dataset, desc="PubMed 재정렬 비교"), start=1,
    ):
        question = sample["question"]
        gold_label = sample["final_decision"].lower()
        gold_pmid = str(sample["pubid"])

        candidate_pmids = list(dict.fromkeys(
            search_pubmed(question, top_k=CANDIDATE_K)
        ))
        fetched = fetch_pubmed_documents(candidate_pmids)
        by_pmid = {document["pmid"]: document for document in fetched}
        candidates = [
            by_pmid[pmid] for pmid in candidate_pmids
            if pmid in by_pmid
            and (by_pmid[pmid]["title"] or by_pmid[pmid]["abstract"])
        ]
        ranked, scores = rerank_documents(
            question, candidates, embedding_model,
        )

        retrieval = {
            "index": index,
            "pubid": gold_pmid,
            "question": question,
            "candidate_pmids": candidate_pmids,
            "native_pmids": [d["pmid"] for d in candidates[:TOP_K]],
            "reranked_pmids": [d["pmid"] for d in ranked[:TOP_K]],
            "reranked_scores": scores[:TOP_K],
            "native_documents": candidates[:TOP_K],
            "reranked_documents": ranked[:TOP_K],
        }
        with trace_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(retrieval, ensure_ascii=False) + "\n")

        for condition, selected in (
            ("pubmed_native_top3", candidates[:TOP_K]),
            ("pubmed_cosine_top3", ranked[:TOP_K]),
            ("pubmed_cosine_top1", ranked[:1]),
        ):
            answer = ask_llm(client, question, selected)
            row = {
                "index": index,
                "pubid": gold_pmid,
                "question": question,
                "gold_label": gold_label,
                "condition": condition,
                "candidate_count": len(candidates),
                "selected_pmids": ",".join(d["pmid"] for d in selected),
                "retrieval_hit": gold_pmid in [d["pmid"] for d in selected],
                **answer,
            }
            row["is_correct"] = (
                row["is_valid"] and row["prediction"] == gold_label
            )
            rows.append(row)
            write_header = not output_path.exists()
            with output_path.open(
                "a", encoding="utf-8-sig", newline=""
            ) as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            print(
                f"\n{index}번 {condition}: {row['prediction']} "
                f"/ 정답 {gold_label} / {row['is_correct']}"
            )

    print("\n" + "=" * 60)
    for condition in dict.fromkeys(row["condition"] for row in rows):
        condition_rows = [r for r in rows if r["condition"] == condition]
        correct = sum(r["is_correct"] for r in condition_rows)
        print(f"{condition}: {correct}/{len(condition_rows)} ({correct / len(condition_rows):.1%})")
    print(f"결과 저장 위치: {output_path}")


if __name__ == "__main__":
    main()
