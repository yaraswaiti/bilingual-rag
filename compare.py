import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import build_chunks

model = SentenceTransformer("intfloat/multilingual-e5-small")

tests = [
    # English (keywords verified against the law text)
    {"q": "How much notice is required to end an open-ended employment contract?", "keyword": "at least one month in advance"},
    {"q": "Within how many days must a worker sue after an arbitrary dismissal?", "keyword": "within sixty days of the date of his dismissal"},
    {"q": "What is the maximum compensation for arbitrary dismissal?", "keyword": "shall not exceed six months"},
    {"q": "How is the end-of-service gratuity calculated?", "keyword": "for each year of his actual service"},
    {"q": "Within how many days must wages be paid?", "keyword": "within a period not exceeding seven days"},
    {"q": "What is the maximum deduction for lost or damaged tools?", "keyword": "does not exceed the wage of five days"},
    {"q": "What is the minimum overtime pay rate?", "keyword": "no less than 125% of his usual wage"},
    {"q": "How much extra is paid for working on holidays or the weekly day off?", "keyword": "no less than (150%) of his usual wage"},
    {"q": "Which day is the weekly day off?", "keyword": "Friday of each week shall be the weekly day off"},
    {"q": "How many days of annual leave is a worker entitled to?", "keyword": "fourteen days of paid annual leave"},
    {"q": "When does annual leave become twenty-one days?", "keyword": "days if he has spent five years in service"},
    {"q": "Is a worker paid for unused annual leave when the service ends?", "keyword": "entitled to receive payment for the days he did not use"},
    {"q": "Can a worker waive his annual leave?", "keyword": "considered null and void"},
    {"q": "How long is sick leave per year?", "keyword": "sick leave of fourteen days during one year"},
    {"q": "Which leave is granted for the Hajj pilgrimage?", "keyword": "To perform the Hajj pilgrimage"},
    {"q": "How long can a woman take unpaid leave to raise her children?", "keyword": "not exceeding one year to devote herself to raising her children"},
    {"q": "How long is unpaid leave to accompany a spouse?", "keyword": "not exceeding two years to accompany their spouse"},
    {"q": "How long is maternity leave?", "keyword": "total of ten weeks"},
    {"q": "When must an employer create an internal work system?", "keyword": "employs ten or more workers"},
    {"q": "What is the maximum paid work suspension per year?", "keyword": "not exceed sixty days in the year"},
    {"q": "Can an employer dismiss a pregnant worker?", "keyword": "starting from the sixth month of her pregnancy"},
    # Arabic
    {"q": "كم مدة الإجازة السنوية للعامل؟", "keyword": "اربعة عشر يوماً"},
    {"q": "متى تصبح الإجازة السنوية 21 يوماً؟", "keyword": "واحداً وعشرين يوماً"},
    {"q": "كم مدة الإجازة المرضية؟", "keyword": "اجازة مرضية"},
    {"q": "هل يجوز لصاحب العمل رفض طلب الإجازة؟", "keyword": "رفض طلب العامل"},
    {"q": "كم مدة إجازة الأمومة؟", "keyword": "المادة 70"},
    {"q": "هل يجوز للعامل التنازل عن إجازته السنوية؟", "keyword": "المادة 64"},
    {"q": "ما هو يوم العطلة الأسبوعية؟", "keyword": "المادة 60"},
    {"q": "ما الإجازة الممنوحة لأداء فريضة الحج؟", "keyword": "المادة 66"},
    {"q": "كم الحد الأقصى لساعات العمل اليومية؟", "keyword": "المادة 56"},
]


def run(chunk_size, overlap):
    chunks = build_chunks(chunk_size=chunk_size, overlap=overlap)
    emb = model.encode(
        ["passage: " + c["text"] for c in chunks],
        normalize_embeddings=True,
        batch_size=32,
    )
    hits = valid = 0
    for t in tests:
        if not any(t["keyword"] in c["text"] for c in chunks):
            continue  # bad keyword: skip, don't count as a miss
        valid += 1
        q = model.encode("query: " + t["q"], normalize_embeddings=True)
        top = np.argsort(-(emb @ q))[:3]
        ok = any(t["keyword"] in chunks[i]["text"] for i in top)
        hits += ok
        if not ok:
            print(f"   missed @{chunk_size}: {t['q']}")
    return hits, valid, len(chunks)


if __name__ == "__main__":
    print(f"{'chunk':>6} {'overlap':>8} {'chunks':>7} {'hit@3':>12}")
    for size, overlap in [(400, 80), (800, 150), (1200, 200)]:
        hits, valid, n = run(size, overlap)
        print(f"{size:>6} {overlap:>8} {n:>7} {hits}/{valid} = {hits / valid:.0%}")