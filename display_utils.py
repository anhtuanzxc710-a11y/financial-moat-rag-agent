import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import re
import unicodedata

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
        with st.expander("View Sources Used in this Section"):
            shown = set()
            for source_doc in sources:
                source = source_doc.metadata
                doc_no = source.get('doc_no')
                if doc_no in shown:
                    continue  # several chunks can come from one source
                shown.add(doc_no)
                title_text = str(source.get('title', 'Untitled Source'))
                url = str(source.get('source', '')).strip()
                published = str(source.get('published', 'N/A')).strip()
                tier = str(source.get('tier', '')).strip()
                chunk_ids = [d.metadata.get('chunk_id') for d in sources if d.metadata.get('doc_no') == doc_no]

                meta_parts = []
                if tier:
                    meta_parts.append(tier)
                if published and published != "N/A":
                    meta_parts.append(published[:10])
                meta_suffix = f" _({' • '.join(meta_parts)})_" if meta_parts else ""

                if url.startswith("http"):
                    st.markdown(f"**[{doc_no}] [{title_text}]({url})**{meta_suffix}")
                else:
                    st.markdown(f"**[{doc_no}] {title_text}**{meta_suffix}")
                    if url:
                        st.caption(f"Source: {url}")
                st.caption("Chunks: " + ", ".join(chunk_ids))
