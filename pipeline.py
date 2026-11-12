import json
import sys
from itertools import permutations
from typing import TypedDict

from langgraph.graph import END, StateGraph

from ingest import collection

Q_WEIGHT = 0.6
MIN_SCORE = 0.15
MAX_ROUNDS = 5


class State(TypedDict):
    jd: str
    questions: list[str]
    cards: dict
    scores: list[dict]
    banned: list[list]
    assignment: list[dict]
    issues: list[str]
    rounds: int
    passed: bool


def analyze(state: State) -> dict:
    col = collection()
    data = col.get(where={"kind": "body"}, include=["metadatas", "documents"])
    cards = {m["card"]: {"title": m["title"], "doc": d} for m, d in zip(data["metadatas"], data["documents"])}
    return {"cards": cards, "banned": [], "issues": [], "rounds": 0}


def retrieve(state: State) -> dict:
    col = collection()
    n = len(state["cards"])
    q_res = col.query(query_texts=state["questions"], n_results=n, where={"kind": "use"})
    jd_res = col.query(query_texts=[state["jd"]], n_results=n, where={"kind": "body"})
    jd_sim = {m["card"]: 1 - d for m, d in zip(jd_res["metadatas"][0], jd_res["distances"][0])}
    # 문항 유사도와 공고 유사도를 섞는다. 문항만 쓰면 직무 연결이 빠지고, 공고만 쓰면 모든 문항이 같은 카드로 몰린다
    scores = [{m["card"]: round(Q_WEIGHT * (1 - d) + (1 - Q_WEIGHT) * jd_sim[m["card"]], 3)
               for m, d in zip(metas, dists)}
              for metas, dists in zip(q_res["metadatas"], q_res["distances"])]
    return {"scores": scores}


def assign(state: State) -> dict:
    banned = {tuple(b) for b in state["banned"]}
    best, best_total = None, -1.0
    for combo in permutations(state["cards"], len(state["questions"])):
        if any((qi, cid) in banned for qi, cid in enumerate(combo)):
            continue
        total = sum(state["scores"][qi][cid] for qi, cid in enumerate(combo))
        if total > best_total:
            best, best_total = combo, total
    if best is None:
        return {"assignment": [], "issues": state["issues"] + ["배정 가능한 조합 없음"]}
    return {"assignment": [{"id": cid, "title": state["cards"][cid]["title"], "score": state["scores"][qi][cid]}
                           for qi, cid in enumerate(best)]}


def review(state: State) -> dict:
    issues, banned = [], list(state["banned"])
    for qi, a in enumerate(state["assignment"]):
        if a["score"] < MIN_SCORE:
            issues.append(f"[반려] Q{qi + 1} {a['id']} 점수 {a['score']} < {MIN_SCORE}, 재배정")
            banned.append([qi, a["id"]])
        if "확인 필요" in state["cards"][a["id"]]["doc"]:
            issues.append(f"[주의] Q{qi + 1} {a['id']} 카드에 '확인 필요' 사실이 남아 있음, 원본 확인 후 사용")
    passed = not any(i.startswith("[반려]") for i in issues)
    return {"issues": state["issues"] + issues, "banned": banned, "passed": passed,
            "rounds": state["rounds"] + 1}


def route(state: State) -> str:
    if state["passed"] or not state["assignment"] or state["rounds"] >= MAX_ROUNDS:
        return END
    return "assign"


def build():
    g = StateGraph(State)
    for name, fn in [("analyze", analyze), ("retrieve", retrieve), ("assign", assign), ("review", review)]:
        g.add_node(name, fn)
    g.set_entry_point("analyze")
    g.add_edge("analyze", "retrieve")
    g.add_edge("retrieve", "assign")
    g.add_edge("assign", "review")
    g.add_conditional_edges("review", route, {"assign": "assign", END: END})
    return g.compile()


graph = build()


def run(jd: str, questions: list[str]) -> dict:
    out = graph.invoke({"jd": jd, "questions": questions})
    return {"assignment": [{"question": q, **a} for q, a in zip(questions, out["assignment"])],
            "issues": out["issues"], "rounds": out["rounds"], "passed": out["passed"]}


if __name__ == "__main__":
    payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
    print(json.dumps(run(payload["jd"], payload["questions"]), ensure_ascii=False, indent=2))
