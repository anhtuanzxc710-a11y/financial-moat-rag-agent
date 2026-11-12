try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
import re
import os
from operator import itemgetter

# Import the tools this agent will use
from tools import get_stock_info, scrape_website


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
    def __init__(self, llm, embeddings_model, search_tool):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.search_tool = search_tool
        self.cache = {}

    def clear_cache(self):
        """Clears the agent's cache."""
        print("Cache cleared.")
        self.cache = {}

    def _search_structured(self, tier, query, max_results=3, use_news=False):
        """Executes search and returns normalized result dicts with title, url, snippet, published."""
        if hasattr(self.search_tool, "text") and hasattr(self.search_tool, "news"):
            items = []
            if use_news:
                try:
                    raw = self.search_tool.news(query, max_results=max_results)
                    for r in raw or []:
                        items.append({
                            "title": r.get("title") or f"{tier} Source",
                            "url": r.get("url") or "",
                            "snippet": r.get("body") or "",
                            "published": r.get("date") or "N/A",
                        })
                except Exception as e:
                    print(f"News search fallback for '{tier}': {e}")
            if not items:
                raw = self.search_tool.text(query, max_results=max_results)
                for r in raw or []:
                    items.append({
                        "title": r.get("title") or f"{tier} Source",
                        "url": r.get("href") or "",
                        "snippet": r.get("body") or "",
                        "published": "N/A",
                    })
            return items

        text_result = self.search_tool.run(query)
        if text_result and "No good search result found" not in text_result:
            return [{
                "title": f"{tier} Results",
                "url": "DuckDuckGo Search",
                "snippet": text_result,
                "published": "N/A",
            }]
        return []

    def _get_context(self, entity_name, ticker):
        """Gathers financial metrics and multi-tiered web/PDF documents for RAG."""
        context_cache_key = f"context_{entity_name}_{ticker}"
        if context_cache_key in self.cache:
            print("Returning cached context.")
            return self.cache[context_cache_key]

        financial_data_result = get_stock_info.run(ticker)
        if financial_data_result.startswith("Error"):
            financial_data = "Financial data is currently unavailable."
        else:
            financial_data = financial_data_result
            print("Successfully collected financial data.")

        source_documents = []
        if not self.search_tool:
            print("--- Web search tool not available, skipping web search. ---")
            self.cache[context_cache_key] = (financial_data, source_documents)
            return financial_data, source_documents

        seen_urls = set()
        tiers = [
            ("Official News & Analysis", f"{entity_name} {ticker} stock news", 4, True),
            ("Critical News & Sentiment", f"{entity_name} {ticker} stock risks lawsuits concerns short interest", 4, False),
            ("Retail Investor Sentiment", f"{entity_name} {ticker} reddit stocks investor sentiment", 4, False),
        ]

        for tier, query, max_res, use_news in tiers:
            print(f"--- Searching for: {tier} ---")
            try:
                results = self._search_structured(tier, query, max_results=max_res, use_news=use_news)
                added = 0
                for item in results:
                    url = item["url"]
                    if "bing.com/aclick" in url:
                        continue
                    content = f"Headline: {item['title']}\nSnippet: {item['snippet']}".strip()
                    if not item["snippet"]:
                        continue
                    if url and url != "DuckDuckGo Search":
                        seen_urls.add(url)
                    source_documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": url or "DuckDuckGo Search",
                            "title": item["title"],
                            "published": item["published"],
                            "tier": tier,
                        },
                    ))
                    added += 1
                print(f"Collected {added} items for {tier}.")
            except Exception as e:
                print(f"An error occurred during web search for tier '{tier}': {e}")

        # Tier 4: Deep Dive (scrape full HTML articles or PDF reports)
        print("--- Searching for: Deep Dive (Articles & PDFs) ---")
        deep_query = f"{entity_name} {ticker} competitive moat financial analysis"
        try:
            deep_results = self._search_structured("Deep Dive", deep_query, max_results=5, use_news=False)
            scraped_count = 0
            for item in deep_results:
                if scraped_count >= 2:
                    break
                url = item["url"]
                if not url or not url.startswith("http") or "bing.com/aclick" in url or url in seen_urls:
                    continue
                seen_urls.add(url)
                print(f"Scraping {url}...")
                scraped = scrape_website.run(url)
                if scraped and not scraped.startswith("Error"):
                    source_documents.append(Document(
                        page_content=scraped,
                        metadata={
                            "source": url,
                            "title": item["title"],
                            "published": item["published"],
                            "tier": "Deep Dive",
                        },
                    ))
                    scraped_count += 1
                    print(f"Successfully scraped deep-dive content from {url}")
                elif item["snippet"]:
                    source_documents.append(Document(
                        page_content=f"Headline: {item['title']}\nSnippet: {item['snippet']}",
                        metadata={
                            "source": url,
                            "title": item["title"],
                            "published": item["published"],
                            "tier": "Deep Dive",
                        },
                    ))
        except Exception as e:
            print(f"An error occurred during Deep Dive search: {e}")

        self.cache[context_cache_key] = (financial_data, source_documents)
        return financial_data, source_documents

    def _create_rag_chain(self, system_prompt, source_documents):
        if not source_documents:
            def no_rag_chain(input_data):
                prompt = ChatPromptTemplate.from_template(system_prompt).format(
                    financial_data=input_data.get("financial_data", "N/A"),
                    context_formatted="No web context was available for this analysis."
                )
                response = self.llm.invoke(prompt)
                return {"answer": response, "sources": []}
            return RunnableLambda(no_rag_chain)

        split_docs = self.text_splitter.split_documents(source_documents)
        vector_store = FAISS.from_documents(documents=split_docs or source_documents, embedding=self.embeddings_model)
        retriever = vector_store.as_retriever(search_kwargs={"k": 6})

        def format_docs_with_citations(docs):
            return "\n\n".join([
                (
                    f"[Source {i+1}]: Title: {doc.metadata.get('title', 'N/A')}\n"
                    f"Published: {doc.metadata.get('published', 'N/A')}\n"
                    f"URL: {doc.metadata.get('source', 'N/A')}\n"
                    f"Content: {doc.page_content}"
                )
                for i, doc in enumerate(docs)
            ])

        rag_chain = (
            {
                "context_docs": itemgetter("input") | retriever,
                "input": itemgetter("input"),
                "financial_data": itemgetter("financial_data"),
            }
            | RunnablePassthrough.assign(context_formatted=lambda x: format_docs_with_citations(x["context_docs"]))
            | { "answer": ChatPromptTemplate.from_template(system_prompt) | self.llm, "sources": itemgetter("context_docs") }
        )
        return rag_chain

    def _run_analysis(self, entity_name, ticker, system_prompt, query_input):
        financial_data, source_documents = self._get_context(entity_name, ticker)
        rag_chain = self._create_rag_chain(system_prompt, source_documents)
        response = rag_chain.invoke({"input": query_input, "financial_data": financial_data})
        if isinstance(response, dict):
             return {"answer": _extract_text(response['answer'].content), "sources": response['sources']}
        else:
             return {"answer": _extract_text(response.content), "sources": []}

    def generate_market_outlook(self, entity_name, ticker):
        print("\nGenerating Market Investor Outlook...")
        system_prompt = (
            "You are a 'Market Investor' analyst. Key financial data:\n{financial_data}\n\n" 
            "Using the retrieved context below, generate a report with these sections:\n"
            "1. **Professional Market Sentiment:** Based on official news/reports.\n"
            "2. **Retail Investor Sentiment:** Based on forum snippets.\n"
            "3. **Valuation Analysis:** Is it expensive or cheap? Reference 'Trailing P/E' and 'Trailing EPS'.\n"
            "**Cite sources with [1], [2] at the end of sentences.**\n\nContext:\n{context_formatted}\n\n"
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
            "**Cite sources with [1], [2] at the end of sentences.**\n\nContext:\n{context_formatted}\n\n"
            "Do not insert line breaks inside words or numbers. Keep paragraphs wrapped normally."
        )
        return self._run_analysis(entity_name, ticker, system_prompt, f"Value analysis for {entity_name}")

    def generate_devils_advocate_view(self, entity_name, ticker):
        print("\nGenerating Devil's Advocate View...")
        system_prompt = (
            "You are a skeptical 'Devil's Advocate' analyst. Your purpose is to challenge the bullish thesis. Key financial data:\n{financial_data}\n\n" 
            "Using the retrieved context below, identify the single strongest counter-argument or hidden risk in a concise paragraph.\n"
            "**Cite the source of your information with [1], [2] at the end of the sentence.**\n\nContext:\n{context_formatted}\n\n"
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
            "2. **Summary Justification:** 3–5 sentences, weighing the three views. No line breaks inside words.\n\n"
            "--- ANALYSIS CONTEXT ---\n{analysis_context}"
        )
        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | self.llm
        response = chain.invoke({"entity_name": entity_name, "analysis_context": combined_analysis})
        return _extract_text(response.content)