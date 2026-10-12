# RFP RAG 프로젝트 — 역할 분담 (`rag/` 파일 담당표)

RFP 문서 100개(PDF 4 · HWP 96)로 질문에 답하는 RAG 챗봇을 **4개 분야**로 나눠 맡습니다.
이 문서는 **누가 어떤 파일을 고치는지**, **어느 노트북 셀을 실행하는지**, **다른 분야에 무엇을 넘기는지**를 정리합니다.

---

## 한눈에 보기

```
 ① 데이터 전처리          ② Retrieval               ③ Generation               ④ 평가
 문서 → 깨끗한 청크  ──▶  청크 → 색인 → 검색  ──▶   질문 해석 → 답변  ──▶     검색 · 생성 · E2E 채점
 parsing                  indexing                  rewrite                    evaluate
 text_utils               retrieval                 llm                        stage_eval
 vision                   pipeline                  credentials                qa_gen
 chunk_split              config                    engine                     async_utils
 notebook_checkpoint      hw
 __init__
```

| 분야 | 한 줄 설명 | 주로 쓰는 노트북 |
|---|---|---|
| ① 데이터 전처리 | 문서를 읽어 검색하기 좋은 청크로 만든다 | `01_rag_build` 0~1장 |
| ② Retrieval | 청크를 색인하고, 질문에 맞는 청크를 찾아온다 | `01_rag_build` 2장 · 02/03 의 검색 설정 |
| ③ Generation | 질문을 해석하고, 찾아온 근거로 답변을 만든다 | `02_generate_eval` 1장 · `03_chatbot` |
| ④ 평가 | 검색·생성·전체(E2E)를 채점하고 어디가 문제인지 찾는다 | `01` 2-5~2-7 · `02` 0-3~0-5, 2장 · `04_e2e_eval` |

---

## ① 데이터 전처리 — 문서 → 깨끗한 청크

| 파일 | 하는 일 |
|---|---|
| `parsing.py` | PDF·HWP·HWPX 로더 호출, 본문·표 추출, 표지·목차 제외 |
| `text_utils.py` | 정제 — 머리글·쪽번호 제거, 끊긴 줄 잇기, 표 중복 제거, 한자 병기 |
| `vision.py` | 그림 → 텍스트 |
| `chunk_split.py` | 청크 분할 — chunk_size, overlap, separators, 섹션 경로 부착 |
| `notebook_checkpoint.py` | 중단 후 이어서 실행, 파싱 품질 표, 최종 파라미터 출력 |
| `__init__.py` | 패키지 표시 (내용 없음) |

- **노트북:** `01_rag_build` — 0-2(파싱·청크 설정) · 0-4(tesseract 점검) · 1-1 ~ 1-4 · **1-2b(그림 처리)** · 3-1(설정 저장)
- **주로 조절하는 값:** `CHUNK_SIZE`, `OVERLAP_RATIO`, `SEPARATORS`, `SECTION_HEADERS`, `PARSING`(표지 제외 쪽 수, 반복 줄 비율 등), `IMG`(OCR · GPT 설정)
- **다른 분야 파일 중 같이 고치는 곳**
  - `config.yaml` / `config.py` 의 `parsing` · `chunking` 섹션 — 실험으로 정한 최종값을 기본값에 반영
  - `pipeline.py` 의 `prepare_chunks` — 청크 방식을 바꿨는데 결과가 그대로면 저장본 재사용 기준(캐시 키)을 확인
- **넘기는 것 → ② Retrieval:** 청크 목록 (`artifacts/` 에 저장)

---

## ② Retrieval — 색인과 검색

| 파일 | 하는 일 |
|---|---|
| `indexing.py` | BM25(kiwi 형태소) 색인, 임베딩(KURE-v1), FAISS 생성·저장 |
| `retrieval.py` | 하이브리드 검색(BM25 × Dense), 멀티 쿼리 병합, 리랭커, 기관 라우팅 |
| `pipeline.py` | 청크 → 색인 → 검색기 → 챗봇 조립 (`prepare_chunks` 부분은 ① 전처리가 수정) |
| `config.py` | 설정 로드와 관리 (각 섹션의 **값**은 해당 분야가 결정) |
| `hw.py` | CPU·GPU 감지, 임베딩 배치·장치 프로필, 메모리 점검 |

- **노트북:** `01_rag_build` 2-1a(BM25) · 2-1b(임베딩·FAISS) · 2-2 · 2-4(리랭커) / `02`·`03` 의 0-2 검색 설정
- **주로 조절하는 값:** `TOP_K`, `DENSE_WEIGHT`(Dense:BM25 비율), `FUSION`(score / rrf), `CANDIDATE_K`, `RERANK`, `RERANK_CANDIDATES`, `ROUTING`, `EMBED_MODEL`, `FAISS_CFG`, `BM25_CFG`
- **참고할 평가:** `01` 의 2-6 · 2-7 — 검색 방식별 Hit@k · MRR, Dense:BM25 비율별 점수와 **추천 DENSE_WEIGHT**
- **넘기는 것 → ③ Generation:** 질문마다 찾아온 근거 청크 (상위 TOP_K 개)

---

## ③ Generation — 질문 해석, 답변, 챗봇

| 파일 | 하는 일 |
|---|---|
| `rewrite.py` | 질문 해석 — 오타·한영 자판·한자·영어 섞인 질문을 검색용 질문으로 바꿈 |
| `llm.py` | 답변 생성 (gpt-5-mini), 프롬프트, 추론 강도, 토큰·비용 집계, 멀티턴 대화 기록 |
| `credentials.py` | OpenAI 키(`.env`), HuggingFace 로그인 |
| `engine.py` | 챗봇 실행 연결부 (루트의 `app.py`, `main.py` 와 함께) — *내용 확인 후 확정* |

- **노트북:** `02_generate_eval` 1-1 ~ 1-3 / `03_chatbot` 전체 (2-4 대본 대화 포함)
- **주로 조절하는 값:** `LLM_MODEL`, `REASONING_EFFORT`, `TEMPERATURE`, `TOP_P`, `MAX_TOKENS`, `ALLOW_GENERAL`, `HISTORY_TURNS`, `QUERY_REWRITE`, `REWRITE_MODEL`, `REWRITE_REASONING_EFFORT`, `REWRITE_N`, `ANSWER_WITH_REWRITE`
- **참고할 평가:** `02` 의 2-2 — 숫자 근거율 · 문장 근거율 · 근거 활용률 · **전제 교정**(틀린 전제 질문을 바로잡았는지) · 함정 유형별 점수
- **넘기는 것 → ④ 평가:** 질문별 답변 · 사용한 근거 · 응답 시간 · 토큰 · 비용

---

## ④ 평가 — 검색 · 생성 · E2E 채점

| 파일 | 하는 일 |
|---|---|
| `evaluate.py` | RAGAS 지표(Faithfulness, Answer Relevancy, Context Precision, Answer Correctness), 골든 데이터셋, 토큰 0 지표 |
| `stage_eval.py` | 단계별 평가 — 검색 Hit@k·MRR, 생성 근거율·전제 교정, E2E 실패 원인 분류 |
| `qa_gen.py` | 평가용 질문·정답 생성 |
| `async_utils.py` | RAGAS 병렬 채점 |

- **노트북:** `01` 2-5 ~ 2-7(검색 평가) / `02` 0-3 ~ 0-5(함정 질문 20개 · 정답 · 핵심 사실), 2 · 2-2 / `04_e2e_eval` 전체
- **관리하는 데이터**
  - `eval/retrieval_golden.jsonl` — 정답 세트 (질문 · 정답 문서 · 근거 키워드). 01 · 02 · 03 · 04 가 같이 씀
  - `02` 0-3 의 `TRAP_SET`(함정 질문 20개) · 0-4 의 `REFERENCES`(정답)
- **결과 파일:** `eval/results/` — `retrieval_01.json` · `generation_02.json` · `chatbot_03_*.json` · `e2e_04.json`
- **넘기는 것 → 전체:** `04` 의 "🛠 고칠 곳" 표 — 실패 원인(🔎 검색 실패 / 🧠 생성 실패 / 📚 근거 없이 정답 / 🙅 근거 있는데 거절)별로 어느 분야가 볼지 정리

---

## 노트북 실행 순서

```
01_rag_build  ──▶  02_generate_eval  ──▶  03_chatbot  ──▶  04_e2e_eval
 ① ② ④(검색)        ③ ④(생성)               ③ ④(멀티턴)       ④(통합)
```

1. **01** — 파싱 · 청크 · 그림(1-2b) → BM25 · 임베딩 · FAISS → 검색 평가(2-5 ~ 2-7) → 설정 저장(3-1)
2. **02** — 함정 질문 20개 답변 생성 → RAGAS → 생성 자체 평가(2-2)
3. **03** — 직접 질문하거나 대본 대화(2-4) 실행 → 턴마다 생성 평가
4. **04** — 01 · 02 · 03 결과를 합쳐 단계별 종합 · 실패 원인 · 02 vs 03 비교 (GPT 비용 $0)

> ⚠️ VM RAM 이 16GB 라서 **노트북은 한 번에 하나만** 실행하고, 끝나면 커널을 종료하세요.

---

## 함께 지킬 규칙

1. **파일 위치·이름을 바꾸지 않습니다.**
   노트북이 `from rag.retrieval import …` 처럼 파일 이름으로 불러오고, 저장된 인덱스(pickle)도 원래 경로를 기억합니다.
   역할별 하위 폴더로 옮기면 노트북이 깨지고 01 을 처음부터 다시 돌려야 할 수 있습니다.
2. **공용 파일은 "내 섹션만" 고칩니다.**
   - `config.py` / `config.yaml` — 파일 관리는 ② Retrieval, 각 섹션(`parsing`, `chunking`, `retrieval`, `llm`, `evaluation`) 값은 해당 분야
   - `pipeline.py` — 파일 관리는 ② Retrieval, `prepare_chunks` 는 ① 전처리
3. **설정을 바꾸면 그 뒤 단계는 다시 실행합니다.**
   - 청크(①)를 바꾸면 → 01 의 2-1a · 2-1b(색인) 다시 → 02 · 03 다시
   - 검색(②)·답변(③) 설정을 바꾸면 → 02 의 결과 폴더가 새로 생겨 답변·RAGAS 를 다시 계산
4. **성능 판단은 평가 결과로 합니다.**
   느낌이 아니라 `01` 2-6(검색) · `02` 2-2(생성) · `04`(E2E) 표의 숫자로 비교하고, 바꾼 설정과 점수를 함께 공유합니다.
5. **비용이 드는 셀을 알고 실행합니다.**
   GPT 비용은 `01` 1-2b(그림 도식 설명) · `02` 1-2(답변) · 2(RAGAS 심판) · `03` 질문마다 발생합니다. 각 셀이 쓴 금액을 출력하니 확인하세요.

---

## (선택) GitHub 리뷰 담당 자동 지정

`.github/CODEOWNERS` 에 넣으면 파일이 바뀔 때 담당자에게 자동으로 리뷰 요청이 갑니다. `@아이디` 를 실제 GitHub 아이디로 바꾸세요.

```
rag/parsing.py rag/text_utils.py rag/vision.py rag/chunk_split.py rag/notebook_checkpoint.py rag/__init__.py  @전처리담당
rag/indexing.py rag/retrieval.py rag/pipeline.py rag/config.py rag/hw.py                                    @Retrieval담당
rag/rewrite.py rag/llm.py rag/credentials.py rag/engine.py                                                  @Generation담당
rag/evaluate.py rag/stage_eval.py rag/qa_gen.py rag/async_utils.py                                          @평가담당
```

---

## 담당자

| 분야 | 담당자 | 연락 |
|---|---|---|
| ① 데이터 전처리 |  |  |
| ② Retrieval |  |  |
| ③ Generation |  |  |
| ④ 평가 |  |  |
