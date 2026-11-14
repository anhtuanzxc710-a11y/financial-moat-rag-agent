"""Replays a saved run and shows how many recent-news sources each persona gets, before vs after.

Usage: python experiments/check_recent_news.py runs/20261114_120337_NVDA.json
Before = generic queries + plain ranking (as in the 12:03 run). After = persona queries + recent-news slots.
No search, no Gemini: only the saved sources and MiniLM.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import rag_pipeline as rp  # noqa: E402
from research_agent import ResearchAgent  # noqa: E402

path = sys.argv[1]
if os.environ.get("MOAT_SELFTEST"):
    from langchain_core.embeddings import DeterministicFakeEmbedding
    emb, tok = DeterministicFakeEmbedding(size=32), None
else:
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    tok = emb._client.tokenizer

agent = ResearchAgent(llm=None, embeddings_model=emb, search_tool=None, tokenizer=tok)
snap = agent.load_snapshot(path)
company, now, ctx = snap["company"], agent.reference_time, agent._context
recent = sorted({d.metadata["doc_no"] for d in ctx["documents"] if rp.is_recent_news(d.metadata, now)})
titles = {d.metadata["doc_no"]: d.metadata["title"] for d in ctx["documents"]}
print(f"{company}: {len(titles)} sources, recent news (<= {rp.RECENT_DAYS} days before {now:%Y-%m-%d}): {recent}")

before_q = [f"Market outlook for {company}", f"Value analysis for {company}", f"Strongest bearish case against {company}?"]
after_q = [f"{company} stock news, analyst ratings, price targets and investor sentiment",
           f"{company} valuation, earnings, margins and competitive moat",
           f"{company} risks: lawsuit, leadership change, scandal, regulation, falling stock price"]
out = []
for persona, qb, qa in zip(["Market", "Value", "Devil"], before_q, after_q):
    vs, n = ctx["vector_store"], len(ctx["chunks"])
    before = rp.select_balanced(vs.similarity_search_with_score(qb, k=n))
    after = rp.select_balanced(rp.prioritize_recent(vs.similarity_search_with_score(qa, k=n), now))
    row = {"persona": persona}
    for name, docs in (("before", before), ("after", after)):
        nos = [d.metadata["doc_no"] for d in docs]
        row[name] = {"sources": nos, "recent_news": [x for x in dict.fromkeys(nos) if x in recent]}
    out.append(row)
    print(f"\n{persona}\n  before: {row['before']['sources']}  recent news: {row['before']['recent_news']}"
          f"\n  after:  {row['after']['sources']}  recent news: {row['after']['recent_news']}")
    for x in row["after"]["recent_news"]:
        print(f"    [{x}] {titles[x][:90]}")
os.makedirs(os.path.join(ROOT, "experiments", "results"), exist_ok=True)
json.dump({"run": os.path.basename(path), "recent": recent, "titles": {str(k): v for k, v in titles.items() if k in recent},
           "personas": out}, open(os.path.join(ROOT, "experiments", "results", "recent_news.json"), "w"), ensure_ascii=False, indent=1)
