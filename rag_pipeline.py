"""Chunking, embedding-input and retrieval helpers for the RAG pipeline.

Pure functions only (no network, no LLM), so every step can be unit-tested and
replayed offline from a saved run snapshot.
"""
import hashlib
import io
import re
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # LangChain 0.x
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Limits -----------------------------------------------------------------
# Only the top-k chunks reach the LLM, so long documents no longer need to be cut
# before splitting. These ceilings only bound download/parse cost.
MAX_HTML_CHARS = 60_000
MAX_PDF_PAGES = 40
MIN_MAIN_CHARS = 500  # an <article>/<main> shorter than this is a teaser, not the body

# MiniLM truncates input at 256 word pieces. Content chunks are kept at 200 tokens
# so the context header (company, tier, title) still fits under the limit.
MODEL_MAX_TOKENS = 256
CHUNK_TOKENS = 200
CHUNK_OVERLAP_TOKENS = 30
TITLE_MAX_CHARS = 80

# Retrieval: final k and the cap on chunks taken from any single source document.
TOP_K = 6
MAX_CHUNKS_PER_DOC = 2

_BLOCK_TAGS = ["h1", "h2", "h3", "h4", "p", "li", "table"]
_NOISE_TAGS = ["script", "style", "noscript", "nav", "aside", "form"]


# --- Extraction (L2, L4) ----------------------------------------------------
def extract_html_text_legacy(html) -> str:
    """Original behaviour: text of <p> tags only (kept for before/after experiments)."""
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))


def _table_to_text(table) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" ; ".join(cells))
    return "\n".join(rows)


def _is_link_list_item(el, text) -> bool:
    """Headings/list items that are only a link: menus, tickers, 'related articles'."""
    link_text = " ".join(a.get_text(" ", strip=True) for a in el.find_all("a"))
    return bool(link_text) and len(link_text) >= 0.8 * len(text)


def _main_content(soup):
    """The page's <article>/<main> when it holds the body text, else the whole page."""
    for root in (soup.find("article"), soup.find("main"), soup.find(attrs={"role": "main"})):
        if root is not None and len(root.get_text(" ", strip=True)) >= MIN_MAIN_CHARS:
            return root
    return soup


def extract_html_text(html) -> str:
    """Headings, paragraphs, list items and tables of the main content, in document order.

    Tables are flattened row by row ("cell ; cell ; cell") so financial figures
    that live in tables survive extraction. Link-only headings and list items
    (menus, ticker strips, related-article lists) are skipped as noise.
    """
    soup = BeautifulSoup(html, "html.parser")
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    for tag in soup(["header", "footer"]):
        if not tag.find_parent(["article", "main"]):  # site chrome; an article's own header keeps its deck
            tag.decompose()
    blocks = []
    summary = (meta.get("content") or "").strip() if meta else ""
    if summary:
        blocks.append(f"Summary: {summary}")
    for el in _main_content(soup).find_all(_BLOCK_TAGS):
        if el.name != "table" and el.find_parent("table"):
            continue  # already covered by the table itself
        if el.name == "li" and el.find("p"):
            continue  # its <p> children are emitted on their own
        text = _table_to_text(el) if el.name == "table" else el.get_text(" ", strip=True)
        if not text or (el.name != "table" and el.name != "p" and _is_link_list_item(el, text)):
            continue
        blocks.append(text)
    return "\n\n".join(blocks)[:MAX_HTML_CHARS]


def extract_pdf_pages(pdf_bytes: bytes):
    """Returns [(page_number, text)] so citations can point to a page."""
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
        for i, page in enumerate(doc):
            if i >= MAX_PDF_PAGES:
                break
            text = page.get_text().strip()
            if text:
                pages.append((i + 1, text))
    return pages


# --- De-duplication (L8) ----------------------------------------------------
def normalize_url(url: str) -> str:
    if not url or not url.startswith("http"):
        return ""
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower().removeprefix("www."), path, "", ""))


def content_hash(text: str) -> str:
    norm = re.sub(r"\W+", " ", (text or "").lower()).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


class Deduplicator:
    """Rejects an item when its URL or its normalised text was already seen."""

    def __init__(self):
        self.urls, self.hashes = set(), set()
        self.rejected = 0

    def accept(self, url: str, text: str) -> bool:
        u, h = normalize_url(url), content_hash(text)
        if (u and u in self.urls) or h in self.hashes:
            self.rejected += 1
            return False
        if u:
            self.urls.add(u)
        self.hashes.add(h)
        return True


# --- Chunking (L5, L6, L7) --------------------------------------------------
def make_splitter(tokenizer=None):
    """Token-based splitter when the embedding tokenizer is available.

    Falls back to the original 1000/100 character splitter if not.
    """
    if tokenizer is not None:
        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=CHUNK_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS
        )
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def context_header(company: str, meta: dict) -> str:
    title = str(meta.get("title", ""))[:TITLE_MAX_CHARS]
    page = f" | p.{meta['page']}" if meta.get("page") else ""
    return f"[{company} | {meta.get('tier', '')} | {title}{page}]"


def build_chunks(documents, company: str, tokenizer=None):
    """Splits documents and gives every chunk an id, a position and a context header.

    Each input Document must carry metadata["doc_no"] (global source number).
    """
    splitter = make_splitter(tokenizer)
    chunks = []
    counters = {}
    for doc in documents:
        start = -1
        for body in splitter.split_text(doc.page_content):
            meta = dict(doc.metadata)
            n = counters.get(meta["doc_no"], 0)
            counters[meta["doc_no"]] = n + 1
            # LangChain's add_start_index mixes token overlap with character offsets, so locate it here.
            start = doc.page_content.find(body, start + 1)
            meta["start_index"] = start
            page = f"p{meta['page']}-" if meta.get("page") else ""
            meta["chunk_id"] = f"S{meta['doc_no']}-{page}c{n}"
            meta["body"] = body
            chunks.append(Document(page_content=f"{context_header(company, meta)}\n{body}", metadata=meta))
    return chunks


# --- Retrieval (L3) ---------------------------------------------------------
def select_balanced(scored_docs, k=TOP_K, max_per_doc=MAX_CHUNKS_PER_DOC):
    """Greedy pick in relevance order, at most `max_per_doc` chunks per source.

    `scored_docs` is [(Document, distance)] sorted best-first. If the cap leaves
    fewer than k chunks, the best remaining ones fill the gap.
    """
    picked, per_doc, overflow = [], {}, []
    for doc, score in scored_docs:
        key = doc.metadata.get("doc_no")
        if per_doc.get(key, 0) < max_per_doc:
            picked.append(doc)
            per_doc[key] = per_doc.get(key, 0) + 1
        else:
            overflow.append(doc)
        if len(picked) == k:
            return picked
    return picked + overflow[: k - len(picked)]


def format_context(docs) -> str:
    """Cites by global source number, so [3] means the same source in every section."""
    return "\n\n".join(
        f"[{d.metadata.get('doc_no')}] (chunk {d.metadata.get('chunk_id')}) Title: {d.metadata.get('title', 'N/A')}\n"
        f"Published: {d.metadata.get('published', 'N/A')}"
        + (f" | Page: {d.metadata['page']}" if d.metadata.get("page") else "")
        + f"\nURL: {d.metadata.get('source', 'N/A')}\nContent: {d.metadata.get('body', d.page_content)}"
        for d in docs
    )
