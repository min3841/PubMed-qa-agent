import argparse
import os
import time
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


MODEL = "gpt-5-nano"
VALID_LABELS = {"yes", "no", "maybe"}

PUBMED_SEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils/esearch.fcgi"
)
PUBMED_FETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils/efetch.fcgi"
)
NCBI_TOOL = "medical_qa_agent"


def parse_args() -> argparse.Namespace:
    # 실행 명령어에서 평가할 문항 수를 입력받는다.
    parser = argparse.ArgumentParser(
        description="PubMed 근거 기반 의료 QA 평가"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="평가할 문항 수 (기본값: 5)",
    )
    return parser.parse_args()


def search_pubmed(
    query: str,
    top_k: int = 3,
) -> list[str]:
    # PubMedQA 질문 하나를 검색어로 사용하여 관련 PMID를 찾는다.
    email = os.getenv("NCBI_EMAIL")

    if not email:
        raise ValueError(
            ".env 파일에 NCBI_EMAIL을 입력하세요."
        )

    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": top_k,
        "tool": NCBI_TOOL,
        "email": email,
    }

    response = requests.get(
        PUBMED_SEARCH_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    return data["esearchresult"]["idlist"]


def fetch_pubmed_xml(pmids: list[str]) -> str:
    # 검색된 PMID에 해당하는 논문 정보를 XML로 가져온다.
    email = os.getenv("NCBI_EMAIL")

    if not email:
        raise ValueError(
            ".env 파일에 NCBI_EMAIL을 입력하세요."
        )

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": NCBI_TOOL,
        "email": email,
    }

    response = requests.get(
        PUBMED_FETCH_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    return response.text


def fetch_pubmed_documents(pmids: list[str]) -> list[dict[str, str]]:
    # 논문을 합치지 않고 각각의 PMID, 제목, 초록을 반환한다.
    if not pmids:
        return []

    xml_text = fetch_pubmed_xml(pmids)
    root = ElementTree.fromstring(xml_text)
    articles = root.findall("PubmedArticle")

    documents: list[dict[str, str]] = []

    for article in articles:
        pmid = article.findtext(".//PMID", default="")
        title_element = article.find(".//ArticleTitle")
        title = (
            "".join(title_element.itertext()).strip()
            if title_element is not None
            else ""
        )

        abstract_elements = article.findall(
            ".//Abstract/AbstractText"
        )
        abstract_parts = []

        for element in abstract_elements:
            label = element.get("Label")
            text = "".join(element.itertext()).strip()

            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)

        abstract = "\n".join(abstract_parts)

        documents.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
            }
        )

    return documents


def build_pubmed_context(pmids: list[str]) -> str:
    # 기존 호출에서는 논문 목록을 하나의 근거 문자열로 합친다.
    if not pmids:
        return "No PubMed evidence was retrieved."

    documents = fetch_pubmed_documents(pmids)
    context_parts = []

    for document in documents:
        context_parts.append(
            f"PMID: {document['pmid']}\n"
            f"Title: {document['title']}\n"
            f"Abstract: {document['abstract']}"
        )

    return "\n\n".join(context_parts)

def run(query: str, top_k: int = 3) -> dict:
    pmids = search_pubmed(
        query=query,
        top_k=top_k,
    )
    context = build_pubmed_context(pmids)

    return {
        "pmids": pmids,
        "context": context,
    }


def run_reranked(
    query: str,
    question: str,
    embedding_model,
    candidate_k: int = 10,
    top_k: int = 3,
) -> dict:
    """PubMed 후보의 제목·초록을 원래 질문과 비교해 상위 문서를 선택한다."""
    from sentence_transformers.util import semantic_search

    if not 1 <= top_k <= candidate_k:
        raise ValueError("1 <= top_k <= candidate_k여야 합니다.")
    if not query.strip() or not question.strip():
        raise ValueError("검색어와 원래 질문은 비어 있을 수 없습니다.")

    # query는 LLM이 만든 검색어, question은 원래 평가 질문이다.
    candidate_pmids = list(dict.fromkeys(search_pubmed(query, top_k=candidate_k)))
    documents = fetch_pubmed_documents(candidate_pmids)

    # XML 반환 순서에 의존하지 않고 PMID로 문서와 검색 결과를 연결한다.
    documents_by_pmid = {document["pmid"]: document for document in documents}
    candidates = []
    candidate_texts = []
    for pmid in candidate_pmids:
        document = documents_by_pmid.get(pmid)
        if document is None:
            continue
        text = f"{document['title']}\n\n{document['abstract']}".strip()
        if not text:
            continue
        candidates.append(document)
        candidate_texts.append(text)

    result = {
        "pmids": [],
        "context": "No PubMed evidence was retrieved.",
        "candidate_pmids": candidate_pmids,
        "ranked_candidates": [],
        "similarity_scores": [],
    }
    if not candidates:
        return result

    question_embedding = embedding_model.encode(
        [question], convert_to_tensor=True, show_progress_bar=False,
    )
    document_embeddings = embedding_model.encode(
        candidate_texts, convert_to_tensor=True, show_progress_bar=False,
    )

    # 기본 점수는 코사인 유사도이며, 모든 후보의 순위를 기록한다.
    hits = semantic_search(
        question_embedding,
        document_embeddings,
        top_k=len(candidates),
    )[0]

    ranked_candidates = []
    context_parts = []
    for rank, hit in enumerate(hits, start=1):
        document = candidates[hit["corpus_id"]]
        score = float(hit["score"])
        selected = rank <= top_k
        ranked_candidates.append({
            **document,
            "rank": rank,
            "score": score,
            "selected": selected,
        })
        if selected:
            result["pmids"].append(document["pmid"])
            result["similarity_scores"].append(score)
            abstract = document["abstract"] or "No abstract available."
            context_parts.append(
                f"PMID: {document['pmid']}\n"
                f"Similarity: {score:.4f}\n"
                f"Title: {document['title']}\n"
                f"Abstract: {abstract}"
            )

    result["context"] = "\n\n".join(context_parts)
    result["ranked_candidates"] = ranked_candidates
    return result


def main() -> None:
    args = parse_args()

    if not 1 <= args.num_samples <= 1000:
        raise SystemExit(
            "--num-samples는 1에서 1000 사이여야 합니다."
        )

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            ".env 파일에 OPENAI_API_KEY를 입력하세요."
        )

    if not os.getenv("NCBI_EMAIL"):
        raise SystemExit(
            ".env 파일에 NCBI_EMAIL을 입력하세요."
        )

    client = OpenAI()

    # Dataset 전체와 질문 하나를 구분한다.
    dataset = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_labeled",
        split="train",
    ).select(range(args.num_samples))

    results: list[dict[str, object]] = []

    # PubMedQA 질문을 하나씩 꺼내 검색, 답변, 평가를 반복한다.
    for index, sample in enumerate(
        tqdm(dataset, desc="PubMed RAG 평가"),
        start=1,
    ):
        question = sample["question"]
        gold_label = sample["final_decision"].lower()

        run_dict = run(query=question, top_k=3)

        pmids = run_dict["pmids"]
        context = run_dict["context"]

        started_at = time.perf_counter()

        response = client.responses.create(
            model=MODEL,
            instructions=(
                "Answer the biomedical question using only "
                "the provided PubMed evidence. "
                "If the evidence is insufficient, answer maybe. "
                "Return exactly one lowercase label: "
                "yes, no, or maybe. "
                "Do not provide an explanation."
            ),
            input=(
                f"Question:\n{question}\n\n"
                f"PubMed evidence:\n{context}"
            ),
        )

        latency_sec = time.perf_counter() - started_at
        prediction = response.output_text.strip().lower()

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
                "retrieved_pmids": ",".join(pmids),
                "retrieved_count": len(pmids),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_sec": round(latency_sec, 3),
            }
        )

        print(f"\n문제: {index}")
        print(f"질문: {question}")
        print(f"정답: {gold_label}")
        print(f"검색된 PMID: {pmids}")
        print(f"모델 답변: {prediction}")
        print(f"정답 여부: {is_correct}")
        print(f"응답 시간: {latency_sec:.3f}초")

    results_df = pd.DataFrame(results)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = (
        results_dir
        / f"pubmed_rag_{args.num_samples}.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("PubMed RAG 평가 완료")
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


if __name__ == "__main__":
    main()
