from rag import search

tests = [
    {"q": "كم مدة الإجازة السنوية للعامل؟", "keyword": "اربعة عشر يوماً عن كل سنة خدمة"},
    {"q": "متى تصبح الإجازة السنوية 21 يوماً؟", "keyword": "واحداً وعشرين يوماً"},
    {"q": "كم مدة الإجازة المرضية؟", "keyword": "اجازة مرضية مدتها اربعة عشر يوماً"},
    {"q": "هل يجوز لصاحب العمل رفض طلب الإجازة؟", "keyword": "ولا يجوز لصاحب العمل رفض طلب العامل"},
    {"q": "What is the notice period for terminating a contract?", "keyword": "notify the other party in writing"},
    {"q": "Who is considered a worker under the law?", "keyword": "Any person, male or female, who performs work in exchange for wages"},
    {"q": "What is this law called?", "keyword": "This law is called (Labor Law of 1996)"},
]

hits = 0
for t in tests:
    top = search(t["q"], k=3)
    found = any(t["keyword"] in c["text"] for c in top)
    hits += found
    print("✅" if found else "❌", t["q"])

print(f"\nHit rate @3: {hits}/{len(tests)} = {hits / len(tests):.0%}")