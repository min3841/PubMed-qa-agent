# PubMedQA 기반 생의학 논문 QA 에이전트

PubMedQA의 생의학 연구 질문에 논문 근거를 바탕으로 `yes`, `no`, `maybe`를 답하는 학습용 프로젝트입니다.
질문만 제공하는 단일 LLM부터 검색 RAG, 도구 선택형 에이전트와 리랭크까지 구현하고 비교했습니다.


## 프로젝트 소개

LLM이 질문을 보고 사용할 검색 도구를 선택한 뒤, 찾은 논문을 근거로 답하는 에이전트를 구현했습니다.
PubMed 키워드 검색과 PubMedQA context의 벡터 검색을 각각 도구로 제공하고,
검색 담당과 최종 판정 담당을 별도의 호출로 분리했습니다.

최종 버전에는 CrossEncoder 리랭크를 적용했습니다.
검색된 후보 중 질문과 관련성이 높은 논문 하나를 선택해 최종 판정에 전달하는 구조입니다.
검색부터 답변까지의 기록을 남겨, 오답이 생기면 논문 검색·선택·판정 중 어느 단계에서 문제가 발생했는지 살펴봤습니다.

## 구현 과정과 블로그 기록

세부 실험 조건, 결과와 오답 분석은 각 블로그 글에 정리했습니다.

- [에이전트 초기 구현](https://own-fb.tistory.com/20)

  벡터 검색과 PubMed 검색을 도구로 등록하고, LLM이 사용할 도구와 검색어를 선택하도록 구현했습니다.

- [PubMed 검색 결과 유사도 재정렬](https://own-fb.tistory.com/21)

  키워드 검색으로 가져온 논문을 코사인 유사도로 다시 선택하는 방식을 비교했습니다.

- [에이전트 평가와 문제 분석](https://own-fb.tistory.com/22)

  전체 평가를 진행하고, 검색 결과와 오답·출력 형식·API 오류를 확인했습니다.

- [검색 담당과 최종 판정 담당 분리](https://own-fb.tistory.com/23)

  검색 문서를 고정한 프롬프트 비교를 바탕으로 두 역할을 분리하고 판정 지침을 정리했습니다.

- [의미 검색과 바이 인코더 정리](https://own-fb.tistory.com/25)

  프로젝트에서 사용한 임베딩 검색의 구조와 특징을 공부했습니다.

- [크로스 인코더와 리랭크 정리](https://own-fb.tistory.com/26)

  질문과 문서를 함께 평가하는 방식과 검색 후 재정렬의 역할을 살펴봤습니다.

- [리랭크 적용 및 5차 트러블슈팅](https://own-fb.tistory.com/27)

  검색 후보를 크로스 인코더로 재정렬하고, 상위 논문 하나만 제공하는 구조를 실험했습니다.

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| `run_baseline.py` · `run_context.py` | 단일 LLM 및 연결된 context 제공 평가 |
| `PM_RAG.py` · `run_vector_rag.py` | PubMed 검색 RAG 및 벡터 검색 RAG |
| `agent_config.py` | 모델 설정, 검색·판정 지침, 도구 정의 |
| `agent_search.py` | 문서 임베딩 준비, 도구 선택과 검색 실행 |
| `run_agent.py` | 리랭크, 최종 판정, 결과 저장과 전체 평가 |

실험 기록은 `results/` 아래의 `results.csv`와 `trace.jsonl`에 보관합니다.

## 사용한 기술

- 개발 환경: Python 3.12, Ubuntu WSL
- 검색·최종 판정 LLM: `gpt-5-nano`
- 임베딩 모델: `NeuML/pubmedbert-base-embeddings`
- 리랭커: `cross-encoder/ms-marco-MiniLM-L6-v2`
- 데이터·검색: [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA), PubMed API

실행에는 본인의 API 키와 환경 설정이 필요하며, 외부 API 사용에 따른 비용이 발생할 수 있습니다.
모델과 데이터는 외부에서 내려받으며, 원 논문·데이터셋·모델의 이용 조건을 따릅니다.
