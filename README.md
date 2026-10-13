# PubMedQA 기반 생의학 논문 QA 에이전트

PubMedQA의 생의학 연구 질문에 논문 근거를 바탕으로 `yes`, `no`, `maybe`를 답하는 학습용 프로젝트입니다. 단일 LLM, 검색 RAG, 도구 선택형 에이전트를 구현하고 비교한 뒤, 검색 후보 수에 따라 근거 문서를 고르는 구조로 발전시켰습니다.

## 최종 에이전트 구조

검색 담당 LLM과 최종 판정 LLM을 별도로 호출합니다. 검색 담당은 질문을 보고 첫 검색 도구와 검색어를 선택하고, 최종 판정 담당은 선택된 논문 한 편만 근거로 받아 답을 고릅니다.

1. 첫 검색이 벡터 검색이면 PubMedQA labeled 문서 1,000개에서 유사한 문서 3개를 찾습니다.
2. 첫 검색이 PubMed 키워드 검색이면 최대 100개의 PMID를 요청합니다. 반환된 PMID가 없거나 논문 내용을 가져오지 못하면 벡터 검색으로 대체합니다. PMID가 100개를 채우면 벡터 검색 문서 3개를 추가합니다.
3. 모인 문서는 PMID로 중복을 제거합니다. 후보가 한 편이면 그대로 사용하고, 여러 편이면 CrossEncoder로 질문·문서 쌍을 재정렬해 1위 문서만 남깁니다.
4. 최종 판정 LLM이 해당 문서를 읽고 `yes`, `no`, `maybe` 중 하나를 출력합니다.

따라서 PubMed 결과가 1편이면 리랭크하지 않고, 2~99편이면 해당 후보를 리랭크합니다. 100편이면 벡터 후보를 합친 뒤 한 번 리랭크합니다.

## 최종 평가

PubMedQA labeled 질문 1,000개를 평가한 결과입니다.

| 지표 | 결과 |
| --- | ---: |
| 정답 PMID 최종 선택률 | 961/1,000 (96.1%) |
| 답변 정확도 | 721/1,000 (72.1%) |
| API 오류 | 0건 |

정답 PMID 최종 선택률은 판정 LLM에 전달한 **문서 한 편의 PMID**가 질문의 정답 PMID와 일치한 비율입니다. 답변 정확도는 전체 1,000문항 중 정답 라벨을 맞힌 비율입니다.

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| `agent_config.py` | 모델, 검색·판정 지침, 검색 도구 및 후보 수 설정 |
| `agent_search.py` | 문서 임베딩 준비, 첫 도구 선택, 검색 결과 수에 따른 추가 검색 |
| `run_agent.py` | PMID 중복 제거, 리랭크, 최종 판정, 문항별 평가·결과 저장 |
| `PM_RAG.py` · `run_vector_rag.py` | PubMed 키워드 검색 및 PubMedQA 벡터 검색 |

## 실행

Python 3.12와 Ubuntu WSL 환경에서 개발했습니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env`에 본인의 `OPENAI_API_KEY`와 `NCBI_EMAIL`을 입력한 뒤 실행합니다. `HF_TOKEN`은 선택 사항입니다.

```bash
python run_agent.py --num-samples 5
# 전체 1,000문항 평가
python run_agent.py --num-samples 1000
```

실행마다 `results/` 아래에 새 폴더가 생기며, 문항별 결과는 `results.csv`, 검색·판정 과정은 `trace.jsonl`에 저장됩니다. `results/`와 `.env`는 Git에 포함하지 않습니다.

## 사용한 기술

- 검색·최종 판정 LLM: `gpt-5-nano`
- 임베딩 모델: `NeuML/pubmedbert-base-embeddings`
- 리랭커: `cross-encoder/ms-marco-MiniLM-L6-v2`
- 데이터·검색: [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA), PubMed API

실행에는 외부 모델 다운로드와 API 호출이 필요하며, API 사용 비용이 발생할 수 있습니다. 데이터셋과 모델의 이용 조건을 확인해 주세요.
