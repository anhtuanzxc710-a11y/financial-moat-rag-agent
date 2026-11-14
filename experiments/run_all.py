"""Before/after experiments for the chunking & embedding fixes (L1-L11).

Step 1 (needs internet, no API key): collect real sources for each ticker with the
improved pipeline and save a snapshot + raw HTML/PDF files.
Step 2 (offline): rebuild BOTH pipelines from the same snapshot - the original
(team version 24db7ca) and the improved one - and measure them with real MiniLM.

Usage (from the project folder):
    python experiments/run_all.py               # collect + measure
    python experiments/run_all.py --skip-collect  # measure existing snapshots only
Outputs: experiments/results/metrics.json and experiments/results/*.png
"""
import argparse
import glob
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "experiments", "data")
OUT = os.path.join(ROOT, "experiments", "results")
TICKERS = [("Apple", "AAPL"), ("NVIDIA", "NVDA"), ("Tesla", "TSLA")]
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import numpy as np  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

import rag_pipeline as rp  # noqa: E402

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

NUM_RE = re.compile(r"\d[\d,.]*\s?(?:%|[BMK]\b|billion|million)?")


def load_model():
    if os.environ.get("MOAT_SELFTEST"):  # offline self-test only: fake model + tiny tokenizer
        from langchain_core.embeddings import DeterministicFakeEmbedding
        return DeterministicFakeEmbedding(size=32), _selftest_tokenizer()
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name=MODEL)
    return emb, emb._client.tokenizer


def _selftest_tokenizer():
    import tempfile
    from transformers import BertTokenizerFast
    path = os.path.join(tempfile.mkdtemp(), "vocab.txt")
    with open(path, "w") as f:
        f.write("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("abcdefghijklmnopqrstuvwxyz0123456789")))
    return BertTokenizerFast(vocab_file=path)


# ---------------------------------------------------------------- collect
def collect(emb, tok):
    from ddgs import DDGS
    from research_agent import ResearchAgent
    for company, ticker in TICKERS:
        print(f"\n=== Collecting {company} ({ticker}) ===")
        agent = ResearchAgent(llm=None, embeddings_model=emb, search_tool=DDGS(), tokenizer=tok,
                              runs_dir=os.path.join(DATA, ticker))
        agent.raw_dir = os.path.join(DATA, ticker, "raw")
        agent._get_context(company, ticker)
        print("Snapshot:", agent.save_run())


# ---------------------------------------------------------------- rebuild both pipelines
def legacy_docs(snap, snap_dir):
    """Team version 24db7ca: no de-dup in tiers 1-3, <p>-only HTML cut at 4000, PDF joined and cut at 8000."""
    import fitz
    docs = []
    for it in snap["search_items"]:
        if it["tier"] == "Deep Dive" or "bing.com/aclick" in it["url"] or not it["snippet"]:
            continue
        docs.append(Document(page_content=f"Headline: {it['title']}\nSnippet: {it['snippet']}",
                             metadata={"source": it["url"], "title": it["title"], "tier": it["tier"]}))
    raw = {int(k): v for k, v in snap.get("raw_files", {}).items() if v}
    done = set()
    for d in snap["documents"]:
        if d["tier"] != "Deep Dive" or d["doc_no"] in done:
            continue
        done.add(d["doc_no"])
        meta = {"source": d["source"], "title": d["title"], "tier": "Deep Dive"}
        path = raw.get(d["doc_no"])
        if path:
            path = _local(path, snap_dir)
            data = open(path, "rb").read()
            if path.endswith(".pdf"):
                with fitz.open(stream=data, filetype="pdf") as pdf:
                    text = "\n".join(p.get_text() for p in pdf).strip()[:8000]
            else:
                text = rp.extract_html_text_legacy(data)[:4000]
        else:
            text = d["content"]
        meta["doc_no"] = d["doc_no"]
        docs.append(Document(page_content=text, metadata=meta))
    for d in docs:
        d.metadata["key"] = d.metadata["source"] + "|" + d.metadata["title"]
    return docs


def _local(path, snap_dir):
    return path if os.path.exists(path) else os.path.join(snap_dir, "raw", re.split(r"[\\/]", path)[-1])


def new_docs(snap):
    return [Document(page_content=d["content"], metadata={k: v for k, v in d.items() if k != "content"})
            for d in snap["documents"]]


def queries(company):
    return [f"Market outlook for {company}", f"Value analysis for {company}",
            f"Strongest bearish case against {company}?"]


def n_tokens(tok, text):
    return len(tok(text, add_special_tokens=True)["input_ids"])


# ---------------------------------------------------------------- measure
def measure(snap_path, emb, tok):
    import fitz
    snap = json.load(open(snap_path, encoding="utf-8"))
    snap_dir = os.path.dirname(snap_path)
    company, ticker = snap["company"], snap["ticker"]
    m = {"ticker": ticker, "company": company, "created_at": snap["created_at"]}

    old_docs, new = legacy_docs(snap, snap_dir), new_docs(snap)
    old_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100,
                                                add_start_index=True).split_documents(old_docs)
    new_chunks = rp.build_chunks(new, company, tok)
    m["sources"] = {"old": len(old_docs), "new": len({d.metadata["doc_no"] for d in new})}
    m["chunks"] = {"old": len(old_chunks), "new": len(new_chunks)}

    # L1 + L2: text kept from each scraped file
    m["scraped"] = []
    for k, path in snap.get("raw_files", {}).items():
        if not path:
            continue
        data = open(_local(path, snap_dir), "rb").read()
        if path.endswith(".pdf"):
            pages = rp.extract_pdf_pages(data)
            full = "\n".join(t for _, t in pages)
            with fitz.open(stream=data, filetype="pdf") as pdf:
                legacy_full = "\n".join(p.get_text() for p in pdf).strip()
            kept_old, cap = legacy_full[:8000], 8000
        else:
            full = rp.extract_html_text(data)
            legacy_full = rp.extract_html_text_legacy(data)
            kept_old, cap = legacy_full[:4000], 4000
        m["scraped"].append({
            "doc_no": int(k), "kind": "pdf" if path.endswith(".pdf") else "html",
            "chars_new": len(full), "chars_old_kept": len(kept_old), "chars_old_before_cap": len(legacy_full),
            "cap": cap, "numbers_new": len(NUM_RE.findall(full)), "numbers_old_kept": len(NUM_RE.findall(kept_old)),
            "numbers_old_p_only_uncut": len(NUM_RE.findall(legacy_full)),
        })

    # L5: tokens actually fed to MiniLM (includes [CLS]/[SEP])
    old_tok = [n_tokens(tok, c.page_content) for c in old_chunks]
    new_tok = [n_tokens(tok, c.page_content) for c in new_chunks]
    m["tokens"] = {
        "old": {"max": max(old_tok), "mean": float(np.mean(old_tok)), "over_256": sum(t > 256 for t in old_tok)},
        "new": {"max": max(new_tok), "mean": float(np.mean(new_tok)), "over_256": sum(t > 256 for t in new_tok)},
    }

    # L11: index build time, 3 builds (old) vs 1 build (new) on the same chunks
    t0 = time.perf_counter(); [FAISS.from_documents(new_chunks, emb) for _ in range(3)]; t3 = time.perf_counter() - t0
    t0 = time.perf_counter(); new_vs = FAISS.from_documents(new_chunks, emb); t1 = time.perf_counter() - t0
    old_vs = FAISS.from_documents(old_chunks, emb)
    m["index_seconds"] = {"old_3_builds": t3, "new_1_build": t1}

    # L3 + L6: retrieval diversity and citation numbering
    m["retrieval"] = []
    old_numbers = {}
    for q in queries(company):
        old_top = old_vs.similarity_search(q, k=6)
        scored = new_vs.similarity_search_with_score(q, k=len(new_chunks))
        new_top = rp.select_balanced(scored)
        uncapped = [d.metadata["doc_no"] for d, _ in scored[:rp.TOP_K]]  # same chunks, no per-source cap
        for i, d in enumerate(old_top):
            old_numbers.setdefault(d.metadata["key"], set()).add(i + 1)
        oc = [d.metadata["key"] for d in old_top]
        nc = [d.metadata["doc_no"] for d in new_top]
        m["retrieval"].append({
            "query": q,
            "old": {"distinct_sources": len(set(oc)), "max_from_one_source": max(oc.count(x) for x in oc)},
            "new": {"distinct_sources": len(set(nc)), "max_from_one_source": max(nc.count(x) for x in nc)},
            "new_without_cap": {"distinct_sources": len(set(uncapped)),
                                "max_from_one_source": max(uncapped.count(x) for x in uncapped)},
        })
    multi = [v for v in old_numbers.values()]
    m["citation_numbers"] = {
        "old_sources_with_several_numbers": sum(len(v) > 1 for v in multi),
        "old_sources_retrieved": len(multi),
    }

    # L8: duplicates that reached the old index vs rejected by the new one
    seen_u, seen_h, dup = set(), set(), 0
    for it in snap["search_items"]:
        if it["tier"] == "Deep Dive" or not it["snippet"]:
            continue
        u, h = rp.normalize_url(it["url"]), rp.content_hash(f"Headline: {it['title']}\nSnippet: {it['snippet']}")
        dup += (u and u in seen_u) or h in seen_h
        seen_u.add(u); seen_h.add(h)
    m["duplicates"] = {"old_indexed": int(dup), "new_rejected": snap.get("dedup_rejected", 0)}

    figures(m, snap, old_chunks, new_chunks, old_tok, new_tok, emb, company)
    return m


# ---------------------------------------------------------------- figures
def figures(m, snap, old_chunks, new_chunks, old_tok, new_tok, emb, company):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = m["ticker"]

    # Token distribution (L5)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.arange(0, max(old_tok + new_tok) + 20, 10)
    ax.hist(old_tok, bins=bins, alpha=.6, label="Bản nhóm: 1.000 ký tự")
    ax.hist(new_tok, bins=bins, alpha=.6, label="Cải tiến: 200 token + tiêu đề ngữ cảnh")
    ax.axvline(256, color="red", ls="--"); ax.text(258, ax.get_ylim()[1] * .9, "giới hạn MiniLM = 256", color="red")
    ax.set_xlabel("Số token đưa vào MiniLM mỗi đoạn"); ax.set_ylabel("Số đoạn"); ax.legend(fontsize=8)
    ax.set_title(f"{t}: độ dài đoạn tính bằng token"); fig.tight_layout(); fig.savefig(f"{OUT}/{t}_tokens.png", dpi=200); plt.close(fig)

    # Chunk map of the longest scraped document (L1 mechanism)
    if m["scraped"]:
        s = max(m["scraped"], key=lambda r: r["chars_new"])
        doc_no = s["doc_no"]
        newc = [c for c in new_chunks if c.metadata["doc_no"] == doc_no]
        offs, spans = 0, []
        for page in sorted({c.metadata.get("page") or 0 for c in newc}):
            pc = [c for c in newc if (c.metadata.get("page") or 0) == page]
            for c in pc:
                spans.append((offs + c.metadata["start_index"], len(c.metadata["body"])))
            offs += max(c.metadata["start_index"] + len(c.metadata["body"]) for c in pc) + 1
        cap = s["cap"]
        olds = [(c.metadata["start_index"], len(c.page_content)) for c in old_chunks if c.metadata.get("doc_no") == doc_no]
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.broken_barh(olds, (12, 6), facecolors=["#4472C4", "#8FAADC"] * 50)
        ax.broken_barh(spans, (2, 6), facecolors=["#C55A11", "#F4B183"] * 500)
        if s["chars_old_before_cap"] > cap:
            ax.axvline(cap, color="red", ls="--"); ax.text(cap, 9.5, f" bản nhóm cắt ở {cap:,} ký tự".replace(",", "."), color="red", fontsize=8)
        ax.set_yticks([5, 15]); ax.set_yticklabels([f"Cải tiến: {len(spans)} đoạn", f"Bản nhóm: {len(olds)} đoạn"])
        ax.set_xlabel("Vị trí ký tự trong tài liệu"); ax.set_title(f"{t}: phủ tài liệu [{doc_no}] ({s['kind'].upper()}, {s['chars_new']:,} ký tự)".replace(",", ".").replace(". ", ", ", 1))
        fig.tight_layout(); fig.savefig(f"{OUT}/{t}_chunk_map.png", dpi=200); plt.close(fig)

    # Embedding space (mechanism) + retrieval picks (L3)
    vecs = np.array(emb.embed_documents([c.page_content for c in new_chunks]))
    qs = queries(company)
    qv = np.array([emb.embed_query(q) for q in qs])
    mu = vecs.mean(0); _, _, vt = np.linalg.svd(vecs - mu, full_matrices=False)
    p, pq = (vecs - mu) @ vt[:2].T, (qv - mu) @ vt[:2].T
    fig, ax = plt.subplots(figsize=(7, 5))
    tiers = sorted({c.metadata["tier"] for c in new_chunks})
    for tier in tiers:
        idx = [i for i, c in enumerate(new_chunks) if c.metadata["tier"] == tier]
        ax.scatter(p[idx, 0], p[idx, 1], s=18, alpha=.7, label=tier)
    vs = FAISS.from_documents(new_chunks, emb)
    pos = {c.metadata["chunk_id"]: i for i, c in enumerate(new_chunks)}
    for j, q in enumerate(qs):
        ax.scatter(pq[j, 0], pq[j, 1], marker="*", s=260, color="black")
        ax.annotate(["Market", "Value", "Devil"][j], pq[j], fontsize=9, xytext=(5, 5), textcoords="offset points")
        top = rp.select_balanced(vs.similarity_search_with_score(q, k=len(new_chunks)))
        for d in top:
            i = pos[d.metadata["chunk_id"]]
            ax.plot([pq[j, 0], p[i, 0]], [pq[j, 1], p[i, 1]], color="grey", lw=.6, alpha=.6)
    ax.legend(fontsize=7, loc="best"); ax.set_xlabel("Thành phần chính 1"); ax.set_ylabel("Thành phần chính 2")
    ax.set_title(f"{t}: không gian vector MiniLM (PCA 2 chiều)")
    fig.tight_layout(); fig.savefig(f"{OUT}/{t}_embedding_space.png", dpi=200); plt.close(fig)


def summary_figure(all_m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [m["ticker"] for m in all_m]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    w = .27
    for i, (key, name) in enumerate([("old", "Bản nhóm"), ("new_without_cap", "Toàn văn, chưa giới hạn"), ("new", "Cải tiến (≤ 2 đoạn/nguồn)")]):
        vals = [max(r[key]["max_from_one_source"] for r in m["retrieval"]) for m in all_m]
        axes[0].bar(x + (i - 1) * w, vals, w, label=name)
    axes[0].set_xticks(x, labels); axes[0].set_title("Số đoạn nhiều nhất từ 1 nguồn trong top-6"); axes[0].legend(fontsize=7)
    old_p = [100 * sum(s["chars_old_kept"] for s in m["scraped"]) / max(1, sum(s["chars_new"] for s in m["scraped"])) for m in all_m]
    axes[1].bar(x - .2, old_p, .4, label="Bản nhóm"); axes[1].bar(x + .2, [100] * len(x), .4, label="Cải tiến")
    axes[1].set_xticks(x, labels); axes[1].set_ylim(0, 110); axes[1].set_title("% văn bản tài liệu cào được giữ lại")
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/summary.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-collect", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    emb, tok = load_model()
    if not args.skip_collect:
        collect(emb, tok)
    results = []
    for _, ticker in TICKERS:
        snaps = sorted(glob.glob(os.path.join(DATA, ticker, "*.json")))
        if not snaps:
            print(f"No snapshot for {ticker}, skipped."); continue
        print(f"\n=== Measuring {ticker} ({os.path.basename(snaps[-1])}) ===")
        results.append(measure(snaps[-1], emb, tok))
    if results:
        summary_figure(results)
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Results in {OUT}")


if __name__ == "__main__":
    main()
