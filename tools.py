import io
import os
import shutil
import tempfile
import certifi


def _ensure_ascii_ca_bundle() -> None:
    """Point curl_cffi to an ASCII CA bundle path when the venv path has non-ASCII chars."""
    ca_path = certifi.where()
    if not ca_path.isascii():
        ascii_ca = os.path.join(tempfile.gettempdir(), "ai_agent_moat_cacert.pem")
        if ascii_ca.isascii():
            if not os.path.exists(ascii_ca):
                shutil.copyfile(ca_path, ascii_ca)
            os.environ.setdefault("CURL_CA_BUNDLE", ascii_ca)
            os.environ.setdefault("SSL_CERT_FILE", ascii_ca)


_ensure_ascii_ca_bundle()

import fitz  # PyMuPDF
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from langchain.tools import tool

@tool
def get_stock_info(ticker: str) -> str:
    """
    Fetches key financial information for a given stock ticker using the yfinance library.
    Uses a multi-step, resilient fallback approach and handles different key casings.
    Returns a formatted string of the data, using 'N/A' for unavailable metrics.
    """
    try:
        # --- FIX: Do not pass a session. Let yfinance handle it with curl_cffi. ---
        stock = yf.Ticker(ticker)

        data = {}
        fi = getattr(stock, "fast_info", {}) or {}

        # --- FIX: Helper function to gracefully handle snake_case and camelCase keys ---
        def pick(*keys):
            for k in keys:
                v = fi.get(k)
                if v is not None:
                    return v
            return None

        data["Current Price"]  = pick("last_price", "lastPrice")
        data["Previous Close"] = pick("previous_close", "previousClose")
        data["Open"]           = pick("open", "regularMarketOpen")
        data["Day Low"]        = pick("day_low", "dayLow")
        data["Day High"]       = pick("day_high", "dayHigh")
        data["Volume"]         = pick("last_volume", "volume")
        data["Market Cap"]     = pick("market_cap", "marketCap")

        # --- FIX: Broader, safer history fallback ---
        if data.get("Current Price") is None:
            try:
                hist = stock.history(period="5d")
                if not hist.empty:
                    close = hist["Close"].dropna()
                    if not close.empty:
                        data["Current Price"] = float(close.iloc[-1])
                    if data.get("Open") is None and "Open" in hist.columns:
                        opening = hist["Open"].dropna()
                        if not opening.empty:
                            data["Open"] = float(opening.iloc[-1])
                    if data.get("Previous Close") is None and len(close) > 1:
                        data["Previous Close"] = float(close.iloc[-2])
            except Exception:
                pass # Ignore history failure

        # Optional: keep get_info(), but never let it fail the run
        try:
            info = stock.get_info()
            data["Company Name"] = info.get('longName')
            data["Symbol"] = info.get('symbol')
            data["Trailing P/E"] = info.get('trailingPE')
            data["Forward P/E"] = info.get('forwardPE')
            data["Trailing EPS"] = info.get('trailingEps')
            data["52 Week High"] = info.get('fiftyTwoWeekHigh')
            data["52 Week Low"] = info.get('fiftyTwoWeekLow')
            summary = info.get('longBusinessSummary')
            if summary:
                data["Business Summary"] = " ".join(str(summary).split())
            if data.get("Market Cap") is None: data["Market Cap"] = info.get('marketCap')
            if data.get("Volume") is None: data["Volume"] = info.get('volume')
        except Exception:
            pass # get_info() is non-critical

        # --- FIX: Remove hard error gate. Proceed with N/A. ---
        if data.get("Current Price") is None:
            data["Current Price"] = "N/A" # Set to N/A instead of returning an error string

        key_order = [
            "Company Name", "Symbol", "Current Price", "Previous Close", "Open",
            "Day Low", "Day High", "Market Cap", "Trailing P/E", "Forward P/E",
            "Trailing EPS", "Volume", "52 Week High", "52 Week Low", "Business Summary"
        ]
        
        report = []
        for key in key_order:
            value = data.get(key)
            report.append(f"{key}: {value if value is not None else 'N/A'}")

        return "\n".join(report)

    except Exception as e:
        return f"Error: An unexpected exception occurred while fetching data for ticker '{ticker}'. Details: {e}"

@tool
def scrape_website(url: str) -> str:
    """Scrapes text from HTML websites and extracts text from PDF files."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()

        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            with fitz.open(stream=io.BytesIO(response.content), filetype="pdf") as doc:
                text = "\n".join(page.get_text() for page in doc)
            text = text.strip()
            if len(text) < 100:
                return f"Error: PDF content from {url} is too short."
            return text[:8000]

        if "text/html" in content_type or "application/xhtml" in content_type:
            soup = BeautifulSoup(response.content, "html.parser")
            text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
            if len(text) < 200:
                return f"Error: HTML content from {url} is too short."
            return text[:4000]

        return f"Error: Unsupported content type '{content_type}' at {url}"
    except requests.RequestException as e:
        return f"Error: Could not access the URL. {e}"
    except Exception as e:
        return f"Error: An unexpected error occurred while processing {url}. {e}"