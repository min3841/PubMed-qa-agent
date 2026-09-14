# PubMed 기반 의료 QA 에이전트

PubMedQA 연구 질문에 대해 `yes`, `no`, `maybe`를 선택하는 학습용 프로젝트입니다.
질문만 제공하는 방식부터 검색 RAG와 도구 선택형 에이전트까지 구현하고 비교했습니다.
임상 판단이나 실제 의료 서비스에 사용하기 위한 프로젝트는 아닙니다.

## 구성

| 파일 | 역할 |
| --- | --- |
| `run_baseline.py` | 질문만 제공하는 단일 LLM 평가 |
| `run_context.py` | 질문에 연결된 PubMedQA context를 직접 제공하는 평가 |
| `PM_RAG.py` | PubMed 검색, 초록 수집, 후보 논문의 유사도 재정렬 |
| `run_vector_rag.py` | PubMedQA context의 임베딩 기반 검색 및 평가 |
| `run_agent.py` | 에이전트 평가 실행 및 문항별 오류 처리 |
| `agent_config.py` | 모델 설정, 검색·최종 판정 지침, 도구 정의 |
| `agent_search.py` | 문서 임베딩 준비, 도구 선택 및 추가 검색 |
| `agent_judge.py` | 검색 대화와 분리된 최종 답변 생성 |
| `agent_results.py` | 실행 설정·검색 기록·CSV 저장 및 결과 요약 |
| `compare_pubmed_rerank.py` | PubMed 기본 순위와 유사도 재정렬 비교 |
| `test.py` | 2번 질문과 문서 3개를 고정한 판정 프롬프트 비교 |

답변 모델은 `gpt-5-nano`, 임베딩 모델은
`NeuML/pubmedbert-base-embeddings`를 사용했습니다.

## 에이전트 동작

1. 검색 담당 LLM이 벡터 검색 또는 PubMed 검색과 검색어를 선택합니다.
2. Python이 요청받은 검색 함수를 실행합니다.
3. 벡터 검색은 결과가 있으면 검색을 종료합니다. 결과가 없을 때만 PubMed로 추가 검색합니다.
4. PubMed를 먼저 사용한 경우에는 LLM이 검색 종료 또는 벡터 추가 검색을 결정합니다.
5. 수집된 근거를 별도의 최종 판정 호출에 전달해 라벨 하나를 생성합니다.

검색은 문항당 최대 2회입니다. PubMed 검색은 후보를 최대 10개 가져와
질문과의 코사인 유사도로 재정렬한 뒤 최대 3개를 제공합니다.
벡터 검색은 PubMedQA context 1,000개에서 상위 3개를 선택합니다.

## 설치 및 실행

개발 환경은 Ubuntu WSL과 Python 3.12입니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

`.env`에 본인의 `OPENAI_API_KEY`와 `NCBI_EMAIL`을 입력합니다.
실행 시 외부 API 요청이 발생하며, OpenAI API 사용량에 따라 비용이 발생합니다.
임베딩 모델은 최초 실행 때 내려받으며 문서 임베딩은 실행할 때마다 생성합니다.

```bash
python run_baseline.py --num-samples 5
python run_context.py --num-samples 5
python PM_RAG.py --num-samples 5
python run_vector_rag.py --num-samples 5
python run_agent.py --num-samples 5
```

에이전트 100문항 평가 및 시작 문항 지정:

```bash
python run_agent.py --num-samples 100
python run_agent.py --num-samples 100 --start-index 56
```

`--start-index`는 평가 시작 번호입니다. 기존 CSV에 이어 쓰지 않고 새 실행 폴더를 만듭니다.
OpenAI·PubMed 요청 오류는 기록한 뒤 다음 문항을 평가합니다.
코드 오류 등 모든 예외를 무시하는 구조는 아닙니다.

## 기록과 실험 결과

저장소에는 구현 포스팅과 최종 비교에 사용한 대표 결과만 포함했습니다.
중간 확인용 실행과 중복 결과는 로컬에만 보존했습니다.

실행별 `results/agent_날짜_시간/` 폴더에 다음 파일을 저장합니다.

- `results.csv`: 문항별 답변, 정답 여부, 검색 기록, 오류 정보
- `trace.jsonl`: 실행 설정, 프롬프트, 검색된 문서와 최종 답변

기존 실험 폴더 일부는 문항 수·변경 사항·정확도를 알 수 있도록 이름을 바꿨습니다.
저장된 로그의 예전 경로는 당시 실행을 기록한 값입니다.

PubMedQA 앞의 100문항 비교 결과:

| 방식 | 정답 수 | 정확도 |
| --- | --- | --- |
| 질문만 제공 | 41 / 100 | 41% |
| PubMed 검색 RAG | 52 / 100 | 52% |
| 연결된 context 직접 제공 | 69 / 100 | 69% |
| 벡터 검색 RAG | 73 / 100 | 73% |

초기 5문항 확인 결과는 질문만 제공 40%, context 직접 제공 60%,
PubMed 검색 RAG 80%, 벡터 검색 RAG 80%였습니다.

포스팅에 사용한 결과는 다음과 같이 정리했습니다.

- `baseline_5.csv`·`context_5.csv`·`pubmed_rag_5.csv`·`pubmed_vector_rag_5.csv`: 초기 5문항
- `baseline_100.csv`·`context_100.csv`·`pubmed_rag_100.csv`·`pubmed_vector_rag_100.csv`: 네 방식 100문항 비교
- `PubMed검색결과_유사도재정렬_실험결과/`: PubMed 검색 후보 재정렬 50문항 비교
- `고정문항_판정프롬프트_비교실험/`: 질문과 검색 문서를 고정한 프롬프트 비교
- 이름에 `통합형에이전트`가 포함된 폴더: 초기 에이전트 결과
- `09_검색판정분리_초기판정지침_5문항_정확도60퍼/`와 `10_검색판정분리_판정지침수정_5문항_정확도100퍼/`: 검색·판정 분리 전후 확인
- 이름이 `최종에이전트_100문항_정확도`로 시작하는 폴더: 개선 후 전체 평가

초기 통합형 에이전트는 100문항 중 1개가 API 오류로 빠졌고,
99개 저장 답변 중 60개가 정답이었습니다. 전체 대상 기준 60%,
답변이 저장된 문항 기준 60.6%입니다.

검색·판정 분리와 지침 수정 후의 전체 실행은 각각 70%, 69%, 68%였습니다.
현재 최종 버전의 저장 결과는 `results/최종에이전트_100문항_정확도68퍼/`입니다.
실행 사이에 검색 처리 조건도 바뀌었으므로 동일 조건 반복 실험으로 해석하지 않습니다.

결과 CSV 일부에는 요약 행이 들어 있습니다. 재집계할 때는 문항 행만 사용해야 합니다.
초기 중단·재개 실행의 CSV 일부는 형식이 깨져 있어 `trace.jsonl`과
`resume_status.json`을 함께 확인해야 합니다. 원본 결과는 보존했습니다.

## 해석 범위

- 벡터 검색 문서에는 평가 질문에 연결된 문서가 포함됩니다.
- 같은 앞의 100문항을 반복 확인했으며, 별도의 미사용 평가 세트 검증은 하지 않았습니다.
- 정답 문서가 검색돼도 최종 라벨은 틀릴 수 있습니다.
- 프롬프트 비교 통제 실험은 2번 질문과 고정된 3개 문서에서 각각 3회 수행했습니다.
- 전체 에이전트 평가에서는 검색을 다시 수행했으므로 프롬프트만의 효과를 분리할 수 없습니다.
- 에이전트가 단일 벡터 RAG보다 높은 정확도를 보장하지 않았습니다.

## 구현 기록

- [1차 트러블슈팅](https://own-fb.tistory.com/20)
- [PubMed 유사도 재정렬 비교](https://own-fb.tistory.com/21)
- [에이전트 100문항 평가와 문제 분석](https://own-fb.tistory.com/22)
- [검색·최종 판정 분리](https://own-fb.tistory.com/23)

데이터셋: [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)

모델과 데이터는 실행 시 외부에서 내려받습니다. 코드의 공개가 원 논문,
데이터셋 또는 모델에 대한 권리를 새로 부여하는 것은 아닙니다.
