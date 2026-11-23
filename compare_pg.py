from compare import tests
from pg_store import connect, search_pg

conn = connect()
all_text = [r[0] for r in conn.execute("SELECT content FROM chunks").fetchall()]

hits = valid = 0
for t in tests:
    if not any(t["keyword"] in c for c in all_text):
        continue  # bad keyword: skip, don't count as a miss
    valid += 1
    top = search_pg(t["q"], k=3)
    ok = any(t["keyword"] in r["text"] for r in top)
    hits += ok
    if not ok:
        print("   missed:", t["q"])

print(f"\npgvector: {len(all_text)} chunks, hit@3 = {hits}/{valid} = {hits / valid:.0%}")