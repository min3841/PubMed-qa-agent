"""벡터 검색 준비 및 LLM의 첫 도구 선택·결과 수 기반 검색 실행."""
import json
from pathlib import Path

from PM_RAG import run as pubmed_search
from run_vector_rag import run as vector_search
from agent_config import (
    MODEL, PUBMED_CANDIDATE_K, RETRIEVAL_TOP_K, SEARCH_INSTRUCTIONS, TOOLS,
)


def append_log(path: Path, record: dict) -> None:
    """실행 과정을 JSONL 한 줄로 즉시 저장한다."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_search_resources(datasets, embedding_model) -> dict:
    """전체 문서 임베딩을 한 번 만들어 모든 질문에서 재사용한다."""
    # 전체 문서와 PMID를 같은 순서로 저장
    documents = []
    pmids = []

    for sample in datasets:
        context_text = "\n\n".join(
            sample["context"]["contexts"]
        )

        documents.append(context_text)
        pmids.append(str(sample["pubid"]))

    # 전체 문서 임베딩은 질문 반복문 밖에서 한 번 생성
    embeddings = embedding_model.encode(
        documents,
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    return {
        "embedding_model": embedding_model,
        "embeddings": embeddings,
        "pmids": pmids,
        "documents": documents,
    }


def run_search_agent(
    client, question: str, search_resources: dict, *,
    index: int, pubid: str, trace_path: Path, search_trace: list,
) -> dict:
    """질문 하나를 검색하고 수집한 근거를 반환한다.

    search_trace는 호출자가 만든 리스트에 검색 직후 추가한다.
    후속 API 호출이 실패해도 완료된 검색 기록은 호출자에게 남는다.
    """
    embedding_model = search_resources["embedding_model"]
    embeddings = search_resources["embeddings"]
    pmids = search_resources["pmids"]
    documents = search_resources["documents"]
    tools = TOOLS
    used_tools = []

    # 1차 호출: 검색 담당 LLM이 첫 번째 도구와 검색어를 선택한다.
    search_response = client.responses.create(
        model=MODEL,
        instructions=SEARCH_INSTRUCTIONS,
        tools=tools,
        input=question,
        tool_choice="required",    # 반드시 도구를 호출
        parallel_tool_calls=False, # 한 응답에서 여러 도구를 동시에 호출하지 않음
    )

    # 도구 선택은 한 번만 맡기고, 후속 검색은 반환 PMID 수로 결정한다.
    for search_index in range(1):
        if search_response.status != "completed": #OpenAI의 응답 생성 상태, completed: 응답 생성 완료
            raise RuntimeError(
                "검색 담당 LLM 응답이 완료되지 않았습니다: "
                f"{search_response.id} "
                f"({search_response.status})"
            )

        tool_calls = [
            item
            for item in search_response.output
            if item.type == "function_call"  # 도구 호출 요청인 경우만
        ]

        if len(tool_calls) != 1:
            raise ValueError(
                "첫 응답에서 검색 도구 하나를 요청해야 합니다."
            )

        item = tool_calls[0]

        # LLM이 만든 함수 입력에서 실제 검색어를 꺼내 검증한다.
        arguments = json.loads(item.arguments)

        if (
            not isinstance(arguments, dict) #isinstance() 어떤 값이 특정 자료형인지 확인
            or not isinstance(arguments.get("query"), str)
            or not arguments["query"].strip()
        ):
            raise ValueError(
                "검색어 query는 비어 있지 않은 문자열이어야 합니다."
            )

        # LLM은 도구를 요청하고, 실제 검색 함수는 파이썬이 실행한다.
        if item.name == "vector_search":
            result = vector_search(
                query=arguments["query"],
                embedding_model=embedding_model,
                embeddings=embeddings,
                pmids=pmids,
                documents=documents,
                top_k=RETRIEVAL_TOP_K,
            )

        elif item.name == "pubmed_search":
            result = pubmed_search(
                query=arguments["query"],
                top_k=PUBMED_CANDIDATE_K,
            )

        else:
            raise ValueError(
                f"알 수 없는 도구: {item.name}"
            )

        used_tools.append(item.name)

        search_record = {
            "search_index": search_index + 1,
            "tool": item.name,
            "query": arguments["query"],
            "response_id": search_response.id,
            "call_id": item.call_id,
            "pmids": result["pmids"],
            "documents": result["documents"],  # 문서별로 구분된 검색 결과
            "context": result["context"],
        }

        if item.name == "pubmed_search":
            search_record["candidate_pmids"] = (
                result["candidate_pmids"]
            )

        # 검색 순서와 각 도구의 원본 결과를 그대로 기록한다.
        search_trace.append(search_record)

        append_log(
            trace_path,
            {
                "event": "search",
                "index": index,
                "pubid": pubid,
                "question": question,
                **search_record,
            },
        )

        print(f"\n문제: {index}")
        print(f"질문: {question}")
        print(f"검색 횟수: {search_index + 1}")
        print(f"선택한 도구: {item.name}")
        print(f"검색어: {arguments['query']}")
        print(f"검색된 PMID: {result['pmids']}")

        if item.name == "pubmed_search":
            print(
                "PubMed 후보 PMID: "
                f"{result['candidate_pmids']}"
            )

        if item.name == "vector_search":
            if not result["documents"]:
                raise ValueError("벡터 검색에서 문서를 찾지 못했습니다.")
            break

        # PubMed의 반환 PMID 수로 분기한다. XML에서 실제로 가져온 문서도 확인한다.
        candidate_count = len(result["candidate_pmids"])
        if candidate_count == 0:
            reason = "keyword_zero_pmids"
        elif not result["documents"]:
            reason = "keyword_no_fetched_documents"
        elif candidate_count == PUBMED_CANDIDATE_K:
            reason = "keyword_full_100_pmids"
        else:
            # 1개는 그대로 전달하고, 2~99개는 최종 단계에서 리랭크한다.
            break

        # 키워드 결과가 없거나 100개를 채우면 원래 질문으로 벡터 3개를 추가한다.
        vector_result = vector_search(
            query=question,
            embedding_model=embedding_model,
            embeddings=embeddings,
            pmids=pmids,
            documents=documents,
            top_k=RETRIEVAL_TOP_K,
        )
        if not vector_result["documents"]:
            raise ValueError("추가 벡터 검색에서 문서를 찾지 못했습니다.")

        vector_record = {
            "search_index": 2,
            "tool": "vector_search",
            "query": question,
            "response_id": None,
            "call_id": None,
            "trigger": reason,
            "pmids": vector_result["pmids"],
            "documents": vector_result["documents"],
            "context": vector_result["context"],
        }
        used_tools.append("vector_search")
        search_trace.append(vector_record)
        append_log(
            trace_path,
            {
                "event": "search",
                "index": index,
                "pubid": pubid,
                "question": question,
                **vector_record,
            },
        )
        print(f"추가 벡터 검색 조건: {reason}")
        print(f"추가 벡터 PMID: {vector_result['pmids']}")
        break

    if not search_trace:
        raise ValueError("검색 결과가 하나도 없습니다.")

    # 최종 선택은 run_agent에서 중복 PMID를 제거한 뒤 수행한다.
    evidence_parts = []

    for record in search_trace:
        evidence_parts.append(
            f"Search {record['search_index']}\n"
            f"Tool: {record['tool']}\n"
            f"Retrieved PMIDs: {record['pmids']}\n"
            f"Evidence:\n{record['context']}"
        )

    collected_context = "\n\n".join(evidence_parts)

    return {
        "used_tools": used_tools,
        "search_trace": search_trace,
        "collected_context": collected_context,
    }
