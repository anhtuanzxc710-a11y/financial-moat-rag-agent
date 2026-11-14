"""Offline tests: no network, no LLM, no model download."""
import json
import os
import sys
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rag_pipeline as rp  # noqa: E402
import research_agent  # noqa: E402

HTML = b"""<html><body><nav>Menu Home About</nav>
<h2>Segment revenue</h2>
<p>Services keep growing.</p>
<table><tr><th>Segment</th><th>FY2025</th></tr>
<tr><td>iPhone</td><td>201.2B</td></tr><tr><td>Services</td><td>96.2B</td></tr></table>
<ul><li>Gross margin 46.2%</li><li>Buyback 90B</li></ul>
<script>var x=1;</script></body></html>"""


def tiny_tokenizer(tmp_path):
    from transformers import BertTokenizerFast
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("abcdefghijklmnopqrstuvwxyz0123456789.,;:%[]|-")
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab))
    return BertTokenizerFast(vocab_file=str(path))


def test_html_extraction_keeps_tables_and_lists():
    new, old = rp.extract_html_text(HTML), rp.extract_html_text_legacy(HTML)
    for needle in ["201.2B", "96.2B", "46.2%", "Segment revenue", "iPhone ; 201.2B"]:
        assert needle in new
        assert needle not in old
    assert "Menu" not in new and "var x" not in new


def test_pdf_pages_are_numbered():
    import fitz
    doc = fitz.open()
    for i in range(3):
        doc.new_page().insert_text((72, 72), f"Page body number {i + 1} " * 5)
    pages = rp.extract_pdf_pages(doc.tobytes())
    assert [p for p, _ in pages] == [1, 2, 3]
    assert "number 2" in pages[1][1]


def test_deduplicator_url_variants_and_content():
    d = rp.Deduplicator()
    assert d.accept("https://www.example.com/a/?utm=1", "Apple beats estimates")
    assert not d.accept("https://example.com/a", "different text")  # same URL
    assert not d.accept("https://other.com/b", "Apple  beats estimates!")  # same text
    assert d.accept("https://other.com/c", "New story")
    assert d.rejected == 2


def _docs():
    long_text = " ".join(f"sentence {i} about apple services margin." for i in range(400))
    return [
        Document(page_content="Headline: short news", metadata={"doc_no": 1, "tier": "Official", "title": "News"}),
        Document(page_content=long_text, metadata={"doc_no": 2, "tier": "Deep Dive", "title": "Report", "page": 3}),
    ]


def test_build_chunks_ids_headers_and_token_limit(tmp_path):
    tok = tiny_tokenizer(tmp_path)
    chunks = rp.build_chunks(_docs(), "Apple", tok)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "S1-c0" and ids[1] == "S2-p3-c0"
    for c in chunks:  # position points at the chunk inside its source text
        src = next(d for d in _docs() if d.metadata["doc_no"] == c.metadata["doc_no"]).page_content
        assert src[c.metadata["start_index"]:].startswith(c.metadata["body"])
    assert all(c.page_content.startswith("[Apple | ") for c in chunks)
    assert max(len(tok.tokenize(c.metadata["body"])) for c in chunks) <= rp.CHUNK_TOKENS


def test_select_balanced_caps_each_source():
    docs = [Document(page_content=str(i), metadata={"doc_no": n}) for i, n in enumerate([1, 1, 1, 1, 2, 3])]
    picked = rp.select_balanced([(d, 0.0) for d in docs], k=4, max_per_doc=2)
    assert [d.metadata["doc_no"] for d in picked] == [1, 1, 2, 3]
    picked = rp.select_balanced([(d, 0.0) for d in docs[:4]], k=3, max_per_doc=2)
    assert len(picked) == 3  # fills from overflow when sources run out


class FakeSearch:
    def news(self, query, max_results=4):
        return [
            {"title": "Apple news", "url": "https://news.com/a", "body": "Apple shares rise.", "date": "2026-11-01"},
            {"title": "Apple news copy", "url": "https://www.news.com/a/", "body": "Copy.", "date": "2026-11-01"},
        ]

    def text(self, query, max_results=4):
        if "moat" in query:
            return [{"title": "Annual report", "href": "https://ir.example.com/ar.pdf", "body": "Report"}]
        return [{"title": "Reddit thread", "href": "https://reddit.com/r/x", "body": "Apple shares rise."}]


def test_agent_offline_end_to_end(tmp_path, monkeypatch):
    calls = {"index": 0}
    real_from_documents = research_agent.FAISS.from_documents

    def counting(*a, **k):
        calls["index"] += 1
        return real_from_documents(*a, **k)

    monkeypatch.setattr(research_agent.FAISS, "from_documents", counting)
    monkeypatch.setattr(research_agent, "get_stock_info", SimpleNamespace(run=lambda t: "Trailing P/E: 39.05"))
    pages = [(1, "Moat page one. " * 300), (2, "Risk page two. " * 300)]
    monkeypatch.setattr(research_agent, "fetch_document", lambda url: {"kind": "pdf", "pages": pages, "raw": b"%PDF"})

    agent = research_agent.ResearchAgent(
        FakeListChatModel(responses=["ok"] * 10), DeterministicFakeEmbedding(size=16), FakeSearch(),
        tokenizer=tiny_tokenizer(tmp_path), runs_dir=str(tmp_path / "runs"),
    )
    agent.raw_dir = str(tmp_path / "raw")
    out = [agent.generate_market_outlook("Apple", "AAPL"), agent.generate_value_analysis("Apple", "AAPL"),
           agent.generate_devils_advocate_view("Apple", "AAPL")]

    assert calls["index"] == 1  # L11: one index per analysis
    docs = agent._context["documents"]
    # L8: URL variant and same snippet text across tiers are dropped
    assert agent.run_log["dedup_rejected"] >= 2
    assert {d.metadata["page"] for d in docs if d.metadata["tier"] == "Deep Dive"} == {1, 2}
    for result in out:
        assert len(result["sources"]) <= rp.TOP_K
        per_doc = {}
        for d in result["sources"]:
            per_doc[d.metadata["doc_no"]] = per_doc.get(d.metadata["doc_no"], 0) + 1
        # cap holds unless every source is already used (then the best leftovers fill k)
        assert max(per_doc.values()) <= rp.MAX_CHUNKS_PER_DOC or len(per_doc) == len({d.metadata["doc_no"] for d in docs})
    # L9: run log is written and can be replayed
    path = agent.save_run({"final_summary": "x"})
    snap = json.load(open(path, encoding="utf-8"))
    assert snap["n_chunks"] == len(agent._context["chunks"]) and len(snap["retrieval"]) == 3
    assert os.path.exists(snap["raw_files"]["3"]) or os.path.exists(snap["raw_files"][3])
    agent.clear_cache()
    agent.load_snapshot(path)
    assert len(agent._context["chunks"]) == snap["n_chunks"]


def test_format_context_uses_global_numbers():
    d = Document(page_content="x", metadata={"doc_no": 7, "chunk_id": "S7-c1", "title": "T", "body": "body", "page": 2})
    text = rp.format_context([d])
    assert text.startswith("[7] (chunk S7-c1)") and "Page: 2" in text and "Content: body" in text


def test_text_search_falls_back_to_news():
    class Blocked(FakeSearch):
        def text(self, query, max_results=4):
            raise RuntimeError("No results found.")
    agent = research_agent.ResearchAgent(None, DeterministicFakeEmbedding(size=8), Blocked())
    items = agent._search_structured("Critical", "Apple risks", max_results=4)
    assert items and all(i["via"] == "news" for i in items)
