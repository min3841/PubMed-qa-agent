"""벡터 검색 준비 및 LLM의 도구 선택·추가 검색 실행."""
import json
from pathlib import Path

from PM_RAG import run_reranked as pubmed_search
from run_vector_rag import run as vector_search
from agent_config import (
    MODEL, PUBMED_CANDIDATE_K, RETRIEVAL_TOP_K, SEARCH_INSTRUCTIONS, TOOLS,
)
from agent_results import append_log


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
        tool_choice="required",
        parallel_tool_calls=False,
    )

    # 첫 검색이 부족한 경우에만 다른 도구로 한 번 더 검색한다.
    for search_index in range(2):
        if search_response.status != "completed":
            raise RuntimeError(
                "검색 담당 LLM 응답이 완료되지 않았습니다: "
                f"{search_response.id} "
                f"({search_response.status})"
            )

        tool_calls = [
            item
            for item in search_response.output
            if item.type == "function_call"
        ]

        # 도구 요청 없이 finish가 오면 현재 근거로 검색을 끝낸다.
        if not tool_calls:
            if not used_tools:
                raise ValueError(
                    "첫 번째 응답에 검색 도구 요청이 없습니다."
                )

            search_decision = (
                search_response.output_text.strip().lower()
            )

            if search_decision != "finish":
                raise ValueError(
                    "검색 종료 응답은 finish여야 합니다: "
                    f"{search_response.output_text!r}"
                )

            break

        if len(tool_calls) != 1:
            raise ValueError(
                "한 번에 도구 하나만 요청해야 합니다."
            )

        item = tool_calls[0]

        # LLM이 만든 함수 입력에서 실제 검색어를 꺼내 검증한다.
        arguments = json.loads(item.arguments)

        if (
            not isinstance(arguments, dict)
            or not isinstance(arguments.get("query"), str)
            or not arguments["query"].strip()
        ):
            raise ValueError(
                "검색어 query는 비어 있지 않은 문자열이어야 합니다."
            )

        if item.name in used_tools:
            raise ValueError(
                f"이미 사용한 도구입니다: {item.name}"
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
                question=question,
                embedding_model=embedding_model,
                candidate_k=PUBMED_CANDIDATE_K,
                top_k=RETRIEVAL_TOP_K,
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
            "context": result["context"],
        }

        if item.name == "pubmed_search":
            search_record["candidate_pmids"] = (
                result["candidate_pmids"]
            )
            search_record["ranked_candidates"] = (
                result["ranked_candidates"]
            )
            search_record["similarity_scores"] = (
                result["similarity_scores"]
            )

        # 첫 번째와 두 번째 검색 결과를 모두 최종 판정용으로 보존한다.
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

            for selected_pmid, score in zip(
                result["pmids"],
                result["similarity_scores"],
            ):
                print(
                    f"선택 논문: {selected_pmid}, "
                    f"코사인 유사도: {score:.4f}"
                )

        # 벡터 검색은 결과가 하나라도 있으면 그 근거를 그대로 사용한다.
        # 결과가 비어 있을 때만 아래 단계에서 PubMed 추가 검색을 허용한다.
        if item.name == "vector_search" and result["pmids"]:
            break

        # 두 번째 검색까지 실행했으면 바로 최종 판정으로 이동한다.
        if search_index == 1:
            break

        # 첫 검색 뒤에는 사용하지 않은 도구만 LLM에 제공한다.
        remaining_tools = [
            tool
            for tool in tools
            if tool["name"] not in used_tools
        ]

        if not remaining_tools:
            break

        tool_output = {
            "pmids": result["pmids"],
            "context": result["context"],
        }

        # 벡터 결과가 비어 있으면 남은 PubMed 도구를 반드시 호출한다.
        # PubMed를 먼저 사용한 경우에는 LLM이 finish 또는 벡터 검색을 고른다.
        next_tool_choice = (
            "required"
            if item.name == "vector_search"
            and not result["pmids"]
            else "auto"
        )

        # 2차 호출: 첫 검색 결과를 보고 finish 또는 다른 도구를 선택한다.
        search_response = client.responses.create(
            model=MODEL,
            previous_response_id=search_response.id,
            instructions=SEARCH_INSTRUCTIONS,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(
                        tool_output,
                        ensure_ascii=False,
                    ),
                }
            ],
            tools=remaining_tools,
            tool_choice=next_tool_choice,
            parallel_tool_calls=False,
        )

    if not search_trace:
        raise ValueError("검색 결과가 하나도 없습니다.")

    # 두 번 검색한 경우 두 도구의 문서를 모두 하나의 근거로 합친다.
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
