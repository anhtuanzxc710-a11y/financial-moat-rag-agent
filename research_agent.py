from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import datetime
import json
import os

# Import the tools this agent will use
from tools import get_stock_info, fetch_document
import rag_pipeline as rp


def _extract_text(content) -> str:
    """Extracts plain markdown text from str or structured content block lists."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content)

class ResearchAgent:
    def __init__(self, llm, embeddings_model, search_tool, tokenizer=None, runs_dir="runs"):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.tokenizer = tokenizer  # embedding tokenizer -> token-based chunking
        self.search_tool = search_tool
        self.runs_dir = runs_dir
        self.raw_dir = None  # set to a folder to keep downloaded HTML/PDF for audits
        self.clear_cache()

    def clear_cache(self):
        """Forgets the current context so the next analysis fetches fresh data."""
        self._context = None
        self.run_log = {}

    def _ddgs(self, kind, tier, query, max_results):
        """One DDGS call (text or news) normalised to title/url/snippet/published; [] on failure."""
        try:
            raw = getattr(self.search_tool, kind)(query, max_results=max_results) or []
        except Exception as e:
            print(f"{kind} search failed for '{tier}': {e}")
            return []
        return [{
            "title": r.get("title") or f"{tier} Source",
            "url": r.get("url" if kind == "news" else "href") or "",
            "snippet": r.get("body") or "",
            "published": r.get("date") or "N/A",
            "via": kind,
        } for r in raw]

    def _search_structured(self, tier, query, max_results=3, use_news=False):
        """Executes search and returns normalized result dicts with title, url, snippet, published.

        Web (text) search can return nothing when engines block automated queries, so each
        tier falls back to the other search type instead of silently dropping the tier.
        """
        if hasattr(self.search_tool, "text") and hasattr(self.search_tool, "news"):
            order = ["news", "text"] if use_news else ["text", "news"]
            for kind in order:
                items = self._ddgs(kind, tier, query, max_results)
                if items:
                    if kind != order[0]:
                        print(f"'{tier}': {order[0]} search empty, used {kind} search instead.")
                    return items
            return []

        text_result = self.search_tool.run(query)
        if text_result and "No good search result found" not in text_result:
            return [{
                "title": f"{tier} Results",
                "url": "DuckDuckGo Search",
                "snippet": text_result,
                "published": "N/A",
            }]
        return []

    # --- Context: collect, de-duplicate, number, chunk, index (once per analysis) ---
    def _add_source(self, docs, pages, item, tier):
        """Adds one source (all its pages) under a new global source number."""
        doc_no = len({d.metadata["doc_no"] for d in docs}) + 1
        for page, text in pages:
            meta = {
                "doc_no": doc_no,
                "source": item["url"] or "DuckDuckGo Search",
                "title": item["title"],
                "published": item["published"],
                "tier": tier,
            }
            if page:
                meta["page"] = page
            docs.append(Document(page_content=text, metadata=meta))
        return doc_no

    def _save_raw(self, doc_no, fetched):
        if not self.raw_dir:
            return None
        os.makedirs(self.raw_dir, exist_ok=True)
        path = os.path.join(self.raw_dir, f"S{doc_no}.{fetched['kind']}")
        with open(path, "wb") as f:
            f.write(fetched["raw"])
        return path

    def _get_context(self, entity_name, ticker):
        """Gathers financial metrics and multi-tiered web/PDF documents, then indexes them."""
        if self._context and self._context["key"] == (entity_name, ticker):
            return self._context

        financial_data_result = get_stock_info.run(ticker)
        if financial_data_result.startswith("Error"):
            financial_data = "Financial data is currently unavailable."
        else:
            financial_data = financial_data_result
            print("Successfully collected financial data.")

        docs, search_log, raw_files = [], [], {}
        dedup = rp.Deduplicator()
        tiers = [
            ("Official News & Analysis", f"{entity_name} {ticker} stock news", 4, True),
            ("Critical News & Sentiment", f"{entity_name} {ticker} stock risks lawsuits concerns short interest", 4, False),
            ("Retail Investor Sentiment", f"{entity_name} {ticker} reddit stocks investor sentiment", 4, False),
        ]
        if not self.search_tool:
            print("--- Web search tool not available, skipping web search. ---")
            tiers = []

        for tier, query, max_res, use_news in tiers:
            print(f"--- Searching for: {tier} ---")
            try:
                results = self._search_structured(tier, query, max_results=max_res, use_news=use_news)
                added = 0
                for item in results:
                    search_log.append({**item, "tier": tier})
                    if "bing.com/aclick" in item["url"] or not item["snippet"]:
                        continue
                    content = f"Headline: {item['title']}\nSnippet: {item['snippet']}"
                    if not dedup.accept(item["url"], content):
                        continue
                    self._add_source(docs, [(None, content)], item, tier)
                    added += 1
                print(f"Collected {added} items for {tier}.")
            except Exception as e:
                print(f"An error occurred during web search for tier '{tier}': {e}")

        # Tier 4: Deep Dive (full HTML articles or PDF reports, no truncation)
        if self.search_tool:
            print("--- Searching for: Deep Dive (Articles & PDFs) ---")
            deep_query = f"{entity_name} {ticker} competitive moat financial analysis"
            try:
                deep_results = self._search_structured("Deep Dive", deep_query, max_results=5, use_news=False)
                scraped_count = 0
                for item in deep_results:
                    search_log.append({**item, "tier": "Deep Dive"})
                    url = item["url"]
                    if scraped_count >= 2 or not url.startswith("http") or "bing.com/aclick" in url:
                        continue
                    if rp.normalize_url(url) in dedup.urls:
                        dedup.rejected += 1
                        continue
                    print(f"Scraping {url}...")
                    try:
                        fetched = fetch_document(url)
                        full_text = "\n".join(t for _, t in fetched["pages"])
                        if not dedup.accept(url, full_text):
                            continue
                        doc_no = self._add_source(docs, fetched["pages"], item, "Deep Dive")
                        raw_files[doc_no] = self._save_raw(doc_no, fetched)
                        scraped_count += 1
                        print(f"Scraped {len(full_text)} characters from {url}")
                    except Exception as e:
                        print(f"Could not scrape {url}: {e}")
                        snippet = f"Headline: {item['title']}\nSnippet: {item['snippet']}"
                        if item["snippet"] and dedup.accept(url, snippet):
                            self._add_source(docs, [(None, snippet)], item, "Deep Dive")
            except Exception as e:
                print(f"An error occurred during Deep Dive search: {e}")

        self._index(entity_name, ticker, financial_data, docs)
        self._context["financial_raw"] = financial_data_result  # with the error text, for the UI
        self.run_log.update({
            "search_items": search_log,
            "dedup_rejected": dedup.rejected,
            "raw_files": raw_files,
        })
        return self._context

    def _index(self, entity_name, ticker, financial_data, docs):
        """Chunks and embeds the sources once; all personas reuse this index."""
        chunks = rp.build_chunks(docs, entity_name, self.tokenizer)
        vector_store = FAISS.from_documents(chunks, self.embeddings_model) if chunks else None
        self._context = {
            "key": (entity_name, ticker),
            "financial_data": financial_data,
            "documents": docs,
            "chunks": chunks,
            "vector_store": vector_store,
        }
        self.run_log = {
            "company": entity_name,
            "ticker": ticker,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "chunking": {
                "mode": "tokens" if self.tokenizer is not None else "characters",
                "chunk_size": rp.CHUNK_TOKENS if self.tokenizer is not None else 1000,
                "top_k": rp.TOP_K,
                "max_chunks_per_doc": rp.MAX_CHUNKS_PER_DOC,
            },
            "financial_data": financial_data,
            "documents": [{**d.metadata, "content": d.page_content} for d in docs],
            "n_chunks": len(chunks),
            "retrieval": {},
        }

    def load_snapshot(self, path):
        """Rebuilds the context from a saved run, so an analysis can be replayed offline."""
        with open(path, encoding="utf-8") as f:
            snap = json.load(f)
        docs = [
            Document(page_content=d["content"], metadata={k: v for k, v in d.items() if k != "content"})
            for d in snap["documents"]
        ]
        self._index(snap["company"], snap["ticker"], snap["financial_data"], docs)
        return snap

    def save_run(self, outputs=None):
        """Writes sources, retrieved chunk ids and outputs to runs/<time>_<ticker>.json."""
        if not self.run_log:
            return None
        if outputs:
            self.run_log["outputs"] = outputs
        os.makedirs(self.runs_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.runs_dir, f"{stamp}_{self.run_log['ticker']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.run_log, f, ensure_ascii=False, indent=2, default=str)
        return path

    # --- Retrieval + generation ---
    def retrieve(self, query):
        """Top-k chunks with at most MAX_CHUNKS_PER_DOC chunks from any one source."""
        ctx = self._context
        if not ctx or not ctx["vector_store"]:
            return []
        # Rank every chunk (a few hundred at most) so the per-source cap never runs dry.
        scored = ctx["vector_store"].similarity_search_with_score(query, k=len(ctx["chunks"]))
        return rp.select_balanced(scored)

    def _run_analysis(self, entity_name, ticker, system_prompt, query_input):
        ctx = self._get_context(entity_name, ticker)
        docs = self.retrieve(query_input)
        context = rp.format_context(docs) if docs else "No web context was available for this analysis."
        prompt = ChatPromptTemplate.from_template(system_prompt).format(
            financial_data=ctx["financial_data"], context_formatted=context
        )
        response = self.llm.invoke(prompt)
        self.run_log.setdefault("retrieval", {})[query_input] = [d.metadata["chunk_id"] for d in docs]
        return {"answer": _extract_text(response.content), "sources": docs}

    def generate_market_outlook(self, entity_name, ticker):
        print("\nGenerating Market Investor Outlook...")
        system_prompt = (
            "You are a 'Market Investor' analyst. Key financial data:\n{financial_data}\n\n" 
            "Using the retrieved context below, generate a report with these sections:\n"
            "1. **Professional Market Sentiment:** Based on official news/reports.\n"
            "2. **Retail Investor Sentiment:** Based on forum snippets.\n"
            "3. **Valuation Analysis:** Is it expensive or cheap? Reference 'Trailing P/E' and 'Trailing EPS'.\n"
            "**Cite sources with the bracketed source numbers shown in the context, e.g. [3], at the end of sentences.**\n\nContext:\n{context_formatted}\n\n"
            "This is an objective summary, not financial advice. "
            "Do not insert line breaks inside words or numbers. Keep paragraphs wrapped normally."
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"Market outlook for {entity_name}")

    def generate_value_analysis(self, entity_name, ticker):
        print("\nGenerating Value Investor Analysis...")
        system_prompt = (
            "You are a 'Value Investor' analyst. Key financial data:\n{financial_data}\n\n" 
            "Using the retrieved context below, generate a business brief with these sections:\n"
            "1. **Valuation Summary:** State if it's 'Overvalued', 'Undervalued', or 'Fairly Valued', justifying with data.\n"
            "2. **SWOT Analysis:** Bulleted Strengths, Weaknesses, Opportunities, Threats.\n"
            "3. **Competitive Moat:** Describe its long-term advantages.\n"
            "**Cite sources with the bracketed source numbers shown in the context, e.g. [3], at the end of sentences.**\n\nContext:\n{context_formatted}\n\n"
            "Do not insert line breaks inside words or numbers. Keep paragraphs wrapped normally."
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"Value analysis for {entity_name}")

    def generate_devils_advocate_view(self, entity_name, ticker):
        print("\nGenerating Devil's Advocate View...")
        system_prompt = (
            "You are a skeptical 'Devil's Advocate' analyst. Your purpose is to challenge the bullish thesis. Key financial data:\n{financial_data}\n\n" 
            "Using the retrieved context below, identify the single strongest counter-argument or hidden risk in a concise paragraph.\n"
            "**Cite the source of your information with the bracketed source numbers shown in the context, e.g. [3], at the end of the sentence.**\n\nContext:\n{context_formatted}\n\n"
            "Do not insert line breaks inside words or numbers. Keep paragraphs wrapped normally."
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"Strongest bearish case against {entity_name}?")

    def generate_final_summary(self, entity_name, market_outlook, value_analysis, devils_advocate):
        print("\nGenerating Final Consensus Summary...")
        combined_analysis = (
            f"--- Market Investor Outlook ---\n{market_outlook}\n\n"
            f"--- Value Investor Analysis ---\n{value_analysis}\n\n"
            f"--- Devil's Advocate View ---\n{devils_advocate}"
        )
        # --- FIX: Replaced prompt with a version that includes a clear rubric ---
        system_prompt = (
            "You are a 'Lead Analyst' synthesizing your team's views for {entity_name}.\n"
            "Return exactly two sections in valid Markdown:\n"
            "1. **Consensus Rating:** Choose exactly one: **Bullish**, **Bearish**, or **Neutral with Caution**.\n"
            "   • Choose **Bullish** if BOTH (A) overall sentiment is net positive AND (B) valuation is not clearly overextended.\n"
            "   • Choose **Bearish** if EITHER (A) major red flags/negative catalysts are present OR (B) valuation is extreme and unsupported.\n"
            "   • Choose **Neutral with Caution** ONLY if evidence is genuinely mixed or insufficient.\n"
            "   • If evidence is mixed but slightly positive, prefer **Bullish**; if mixed but slightly negative, prefer **Bearish**.\n"
            "2. **Summary Justification:** 3–5 sentences, weighing the three views. No line breaks inside words.\n"
            "Keep the analysts' bracketed source numbers (e.g. [3]) after each point you take from them.\n\n"
            "--- ANALYSIS CONTEXT ---\n{analysis_context}"
        )
        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | self.llm
        response = chain.invoke({"entity_name": entity_name, "analysis_context": combined_analysis})
        return _extract_text(response.content)