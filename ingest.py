import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from fastembed import TextEmbedding

VAULT = Path(r"C:\Users\lyt0076\Documents\obsidian-job-vault\중고신입")
DB_DIR = Path(__file__).parent / "db"
MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# 카드 본문에는 "성장과정", "협업" 같은 문항 단어가 없어 검색이 안 걸린다. 기존 STAR 표의 주 용도를 붙인다.
USAGE = {
    "A01": "지원동기, 직무역량, 현업 성과, 소재 평가, 공정 조건",
    "STAR01": "직무역량, 지원동기, 전문지식, 불량 분석, 문제 해결",
    "STAR02": "직무역량 보조, 강점, 꼼꼼함, 신뢰성 평가",
    "STAR03": "지원동기, 전문지식, 소재 공정 직무 적합성, 계면 접합",
    "STAR04": "성장과정, 가치관, 영향을 준 사건, 리더십, 표준화",
    "STAR05": "협업, 갈등 해결, 학업 외 활동, 팀 프로젝트, 입사 후 포부",
}


class FastEmbedFn(EmbeddingFunction):
    def __init__(self):
        self.model = TextEmbedding(MODEL)

    def __call__(self, input):
        return [v.tolist() for v in self.model.embed(list(input))]


def collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection("cards", embedding_function=FastEmbedFn(),
                                           metadata={"hnsw:space": "cosine"})


def strip_frontmatter(text):
    return re.sub(r"\A---.*?---\s*", "", text, flags=re.S)


def load_cards():
    cards = []
    for f in sorted((VAULT / "20_기여카드").rglob("*.md")):
        raw = f.read_text(encoding="utf-8")
        if "공개가능: false" in raw or f.stem.endswith("쌓을거리"):
            continue
        body = strip_frontmatter(raw)
        title = re.search(r"^# (.+)$", body, re.M)
        cards.append({"id": f.stem.split("_")[0], "title": title.group(1) if title else f.stem,
                      "source": "엘티 기여카드", "text": body})

    # 옛 STAR_경험카드.md는 폐기·미보유 자격(JLPT) 카드가 섞여 있어, 확정본인 몰입캠프 통합본 5절만 쓴다
    doc = (VAULT / "40_기존자산" / "몰입캠프_경험정리_통합_이윤태.md").read_text(encoding="utf-8")
    section = doc.split("# 5. STAR", 1)[1].split("\n# 6.", 1)[0]
    for block in re.split(r"^## ", section, flags=re.M)[1:]:
        head, _, rest = block.partition("\n")
        m = re.match(r"STAR (\d+) — (.+)", head)
        if m:
            cards.append({"id": f"STAR{m.group(1)}", "title": m.group(2).strip(),
                          "source": "STAR 5건", "text": rest})
    return cards


if __name__ == "__main__":
    cards = load_cards()
    chromadb.PersistentClient(path=str(DB_DIR)).delete_collection("cards") if DB_DIR.exists() else None
    col = collection()
    # 문항은 "어떤 용도의 경험인가"와, 공고는 "무슨 일을 했나"와 맞춰야 해서 카드마다 벡터를 두 개 둔다
    for kind, doc in [("use", lambda c: USAGE.get(c["id"], c["title"])),
                      ("body", lambda c: f"{c['title']}\n{c['text']}")]:
        col.add(ids=[f"{c['id']}#{kind}" for c in cards],
                documents=[doc(c) for c in cards],
                metadatas=[{"card": c["id"], "kind": kind, "title": c["title"], "source": c["source"]} for c in cards])
    for c in cards:
        print(c["id"], c["title"], file=sys.stdout)
    print(f"indexed {len(cards)} cards ({col.count()} vectors)")
