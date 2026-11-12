# job-rag

자소서 문항과 공고를 넣으면, Obsidian vault의 경험 카드 중 문항별로 쓸 카드를 겹치지 않게 골라 주는 도구.

## 구성
- `ingest.py`: vault의 기여카드(공개가능)와 STAR 5건을 읽어 Chroma 벡터 DB에 넣는다. 카드마다 "용도" 벡터와 "본문" 벡터 2개.
  임베딩은 fastembed `paraphrase-multilingual-mpnet-base-v2`(로컬 ONNX, API 키 불필요).
- `pipeline.py`: LangGraph 4노드 `analyze → retrieve → assign → review`.
  - retrieve: 점수 = 0.6 × (문항 ↔ 용도 유사도) + 0.4 × (공고 ↔ 본문 유사도)
  - assign: 문항 수만큼 카드 조합을 전부 보고 점수 합이 최대인 조합 선택(카드 중복 없음)
  - review: 점수 < MIN_SCORE면 반려하고 assign으로 되돌림(최대 5회). 카드에 "확인 필요"가 남아 있으면 주의 표시.
- `api.py`: FastAPI `POST /assign {jd, questions}`

## 실행
```
.venv\Scripts\python.exe ingest.py
.venv\Scripts\python.exe pipeline.py samples\semes.json
.venv\Scripts\python.exe -m uvicorn api:app --port 8799
```

## 검증 (2026-11-12)
세메스 공정개발 4문항: 실제 제출본 배정과 3/4 일치(Q2 매장, Q3 Probe Card, Q4 저온솔더). Q1은 실제 A01, 도구는 STAR01.

## 하지 않은 것
- LLM으로 문장 생성은 하지 않는다(검색과 규칙 검토만). n8n, Slack 연동 없음.
