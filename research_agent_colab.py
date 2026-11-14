"""Colab entry point: the same RAG pipeline as the Streamlit app (research_agent.py),
searched through Google Programmable Search instead of DDGS."""
import os

try:
    from langchain_google_community import GoogleSearchAPIWrapper
except ImportError:  # removed from langchain-community in 0.4
    from langchain_community.utilities import GoogleSearchAPIWrapper

import research_agent


class GoogleSearch:
    """Gives GoogleSearchAPIWrapper the text()/news() interface the agent uses for DDGS."""

    def __init__(self, wrapper):
        self.wrapper = wrapper

    def text(self, query, max_results=4):
        return [
            {"title": r.get("title"), "href": r["link"], "url": r["link"],
             "body": r.get("snippet", ""), "date": r.get("publication_time")}
            for r in self.wrapper.results(query, num_results=max_results)
            if "link" in r  # the wrapper returns [{"Result": "No good ..."}] when empty
        ]

    news = text  # Programmable Search has no separate news endpoint


class ResearchAgent(research_agent.ResearchAgent):
    def __init__(self, llm, embeddings_model, runs_dir="runs"):
        wrapper = GoogleSearchAPIWrapper(
            google_cse_id=os.environ.get("GOOGLE_CSE_ID"),
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
        )
        tokenizer = getattr(getattr(embeddings_model, "_client", None), "tokenizer", None)
        super().__init__(llm, embeddings_model, GoogleSearch(wrapper), tokenizer=tokenizer, runs_dir=runs_dir)
