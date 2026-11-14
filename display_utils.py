import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import re
import unicodedata

from rag_pipeline import split_by_citation

# --- FIX: Added a helper function to sanitize text before rendering ---
def _clean_markdown(text: str) -> str:
    """Normalizes and cleans markdown text to prevent rendering artifacts."""
    if not isinstance(text, str):
        return str(text)
    # Normalize unicode characters
    s = unicodedata.normalize("NFKC", text)
    # Replace single-character line breaks that occur inside words/numbers
    s = re.sub(r'(?<=\w)\n(?=\w)', ' ', s)
    # Collapse 3 or more newlines into a maximum of 2
    s = re.sub(r'\n{3,}', '\n\n', s)
    # Collapse excessive spaces or tabs into a single space
    s = re.sub(r'[ \t]{2,}', ' ', s)
    # Streamlit reads $...$ as LaTeX, which garbles prices such as "$356.27 ... $360.83"
    s = re.sub(r'(?<!\\)\$', r'\\$', s)
    return s.strip()

def display_analysis(title, company_name, result, is_summary=False):
    """
    Formats and displays the agent's analysis in a structured Streamlit format.
    """
    st.subheader(title)

    answer = result if is_summary else result.get('answer', 'No analysis generated.')
    # --- FIX: Clean the answer text before displaying it ---
    answer = _clean_markdown(answer)
    
    sources = result.get('sources', []) if not is_summary else []

    with st.container(border=True):
        st.markdown(answer)

    # Use an expander for the sources to keep the UI clean
    if sources:
        cited, not_cited = split_by_citation(answer, sources)
        with st.expander(f"View Sources Used in this Section ({len(cited)} cited, {len(not_cited)} retrieved but not cited)"):
            for group, note in ((cited, ""), (not_cited, " · retrieved, not cited")):
                for source_doc in group:
                    source = source_doc.metadata
                    doc_no = source.get('doc_no')
                    title_text = str(source.get('title', 'Untitled Source'))
                    url = str(source.get('source', '')).strip()
                    published = str(source.get('published', 'N/A')).strip()
                    tier = str(source.get('tier', '')).strip()
                    chunk_ids = [d.metadata.get('chunk_id') for d in sources if d.metadata.get('doc_no') == doc_no]

                    meta_parts = [p for p in (tier, published[:10] if published != "N/A" else "") if p]
                    meta_suffix = f" _({' • '.join(meta_parts)}{note})_" if meta_parts or note else ""
                    link = f"[{title_text}]({url})" if url.startswith("http") else title_text
                    st.markdown(f"**[{doc_no}] {link}**{meta_suffix}")
                    if url and not url.startswith("http"):
                        st.caption(f"Source: {url}")
                    st.caption("Chunks: " + ", ".join(chunk_ids))
