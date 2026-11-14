"""Which DDGS engines answer from this machine? Prints one line per engine.

Usage: python experiments/check_search.py
"""
from ddgs import DDGS

QUERY = "Apple AAPL stock risks lawsuits concerns"
ddgs = DDGS()
for kind, engines in [("text", ["duckduckgo", "brave", "yahoo", "mojeek", "startpage", "wikipedia", "grokipedia"]),
                      ("news", ["duckduckgo", "bing", "yahoo"])]:
    for engine in engines:
        try:
            n = len(getattr(ddgs, kind)(QUERY, max_results=5, backend=engine) or [])
            print(f"{kind:5} {engine:11} {n} results")
        except Exception as e:
            print(f"{kind:5} {engine:11} FAILED: {str(e)[:80]}")
