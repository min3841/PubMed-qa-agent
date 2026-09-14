import os
import argparse
import time
import pandas as pd

from openai import OpenAI
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer  #임베딩 모델을 불러와 텍스트를 벡터로 변환하기 위해 사용
from sentence_transformers.util import semantic_search #벡터 사이의 유사도를 계산하고 관련 관련 문서를 검색

MODEL = "gpt-5-nano"
LABEL = {"yes", "no", "maybe"}

def parser()-> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="벡터 RAG 평가"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="문항 평가 기본 : 5"
    )
    return parser.parse_args()

def run(query, embedding_model, embeddings, pmids, documents, top_k=3):

    query_embedding = embedding_model.encode(
    [query],
    convert_to_tensor=True,
    )

    hits = semantic_search(
        query_embedding,   # 질문을 숫자로 변환한 벡터
        embeddings,        # 모든 문서의 벡터
        top_k=top_k,           # 유사도가 높은 문서 3개 선택
    )[0]

    retrieved_Context  = []
    retrieved_pmids = []

    for hit in hits:
        corpus_id = hit["corpus_id"]  # semantic_search()의 반환 형식 키
        score = hit["score"]

        # 같은 위치에 저장된 PMID와 문서 가져오기
        retrieved_pmid = pmids[corpus_id]
        retrieved_document = documents[corpus_id]

        retrieved_pmids.append(retrieved_pmid)

        # LLM에 전달할 문서 목록에 추가
        retrieved_Context.append(
            f"PMID : {retrieved_pmid}\n"
            f"Similarity : {score:.4f}\n"
            f"Context : {retrieved_document}"
        )

    return {
        "pmids": retrieved_pmids,
        "context": "\n\n".join(retrieved_Context),
    }
    

def main()-> None:
    arg = parser()

    if not 1 <= arg.num_samples <= 100:
        raise SystemExit("num_samples는 1부터 100 사이여야 합니다.")

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("api키가 존재하지 않습니다.")

    client = OpenAI()

    embedding_model = SentenceTransformer(
    "NeuML/pubmedbert-base-embeddings"
)

    datasets = load_dataset(
            "qiaojin/PubMedQA",
            "pqa_labeled",
            split="train",
    )
    documents = []
    pmids = []

    for sample in datasets:
        context_text = "\n\n".join(
            sample["context"]["contexts"]
        )

        documents.append(context_text)
        pmids.append(str(sample["pubid"]))
        

    embeddings = embedding_model.encode(
        documents,                             # 벡터로 변환할 문서 목록
        convert_to_tensor=True,                # 결과를 PyTorch Tensor로 반환
        show_progress_bar=True,                # 진행률 표시
    )

    num_dataset = datasets.select(range(arg.num_samples))

    results: list[dict[str, object]] = []

    for index, sample in enumerate(
        tqdm(num_dataset, desc="벡터 RAG 평가"),
        start=1,
    ):

        question = sample["question"]
        gold_label = sample["final_decision"]
        gold_pmid = str(sample["pubid"])

        retrieved = run(
        query=question,
        embedding_model=embedding_model,
        embeddings=embeddings,
        documents=documents,
        pmids=pmids,
        )

        retrieved_pmids = retrieved["pmids"]
        context = retrieved["context"]

        retrieval_hit = gold_pmid in retrieved_pmids

        if retrieval_hit:
            retrieval_rank = (
               retrieved_pmids.index(gold_pmid) + 1 
            )
        else:
            retrieval_rank = None

        print(f"\n\n질문: {question}")
        print(f"정답 문서 검색 성공: {retrieval_hit}")
        print(f"정답 문서 검색 순위: {retrieval_rank}")

        started_at = time.perf_counter()

        response = client.responses.create(
            model=MODEL,
            instructions=(
                "Answer the biomedical research question using only "
                "the provided evidence. "
                "If the evidence is insufficient, answer maybe. "
                "Return exactly one lowercase label: "
                "yes, no, or maybe. "
                "Do not provide an explanation."
            ),
            input=(
                f"Question:\n{question}\n\n"
                f"Retrieved evidence:\n{context}"
            ),
        )

        latency_sec = (
            time.perf_counter() - started_at
        )

        prediction = (
            response.output_text.strip().lower()
        )

        is_valid = prediction in LABEL

        is_correct = (
            is_valid
            and prediction == gold_label.lower()
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = response.usage.total_tokens

        results.append(
            {
                "index": index,
                "pubid": gold_pmid,
                "retrieved_pmids": ",".join(retrieved_pmids),
                "retrieval_hit": retrieval_hit,
                "retrieval_rank": retrieval_rank,
                "retrieved_context_count": len(retrieved_pmids),
                "question": question,
                "gold_label": gold_label,
                "prediction": prediction,
                "is_valid": is_valid,
                "is_correct": is_correct,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_sec": round(latency_sec, 3),
            }
        )

        print(f"\n문제: {index}")
        print(f"질문: {question}")
        print(f"정답: {gold_label}")
        print(f"검색된 PMID: {retrieved_pmids}")
        print(f"모델 답변: {prediction}")
        print(f"정답 여부: {is_correct}")
        print(f"응답 시간: {latency_sec:.3f}초")

    results_df = pd.DataFrame(results)

    hit_at_1 = (
        results_df["retrieval_rank"]
        .fillna(float("inf"))       # 검색 실패값 NaN을 무한대로 변경
        .eq(1)
        .mean()
    )

    hit_at_3 = (
        results_df["retrieval_rank"]
        .fillna(float("inf"))
        .le(3)  # 값 <= 3 (ge:이상, lt:미만, gt:초과 )
        .mean()
    )

    mrr = (
        results_df["retrieval_rank"]
        .apply(
            lambda rank: ( 
                0.0
                if pd.isna(rank)  #값이 없는지 검사
                else 1.0 / rank
            )
        )
        .mean()
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = (
        results_dir
        / f"pubmed_vector_rag_{arg.num_samples}.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("PubMed vector_RAG 평가 완료")
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
    print(f"Hit@1: {hit_at_1:.1%}")
    print(f"Hit@3: {hit_at_3:.1%}")
    print(f"MRR: {mrr:.3f}")
    print(f"결과 저장 위치: {output_path}")


if __name__ == "__main__":
    main()
