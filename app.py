import os
import time
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import RSSFeedLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()


# -------------------------
# Helpers
# -------------------------
def build_vectorstore_from_medium_rss(
    rss_urls: List[str],
    openai_api_key: str,
    max_items_per_feed: int = 25,
) -> FAISS:
    """
    Loads articles via RSS, splits them, and indexes into FAISS.
    Keeps Medium link/title in metadata for source display.
    """
    loader = RSSFeedLoader(urls=rss_urls)
    docs = loader.load()

    # Optional: cap volume to keep it “simple and fast”
    docs = docs[: max_items_per_feed * max(1, len(rss_urls))]

    # Ensure link/title exist in metadata (RSSFeedLoader generally includes these, but we normalize)
    for d in docs:
        md = d.metadata or {}
        d.metadata = {
            "source": md.get("link") or md.get("source") or "",
            "title": md.get("title") or md.get("title_detail") or "Medium Article",
            "published": md.get("published") or md.get("pubDate") or "",
        }

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs


def format_sources(retrieved_docs) -> List[Dict[str, Any]]:
    """
    Deduplicate sources by URL and return a compact list.
    """
    seen = set()
    sources = []
    for d in retrieved_docs:
        url = (d.metadata or {}).get("source", "")
        title = (d.metadata or {}).get("title", "Medium Article")
        published = (d.metadata or {}).get("published", "")
        if url and url not in seen:
            seen.add(url)
            sources.append({"title": title, "url": url, "published": published})
    return sources


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Medium RAG Chatbot", layout="wide")
st.title("Medium RAG Chatbot (LangChain + OpenAI)")

with st.sidebar:
    st.header("Setup")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
    )

    st.caption("Enter one RSS URL per line (profiles/publications/topics).")
    rss_text = st.text_area(
        "Medium RSS feed URLs",
        value="https://medium.com/feed/@TowardsDataScience\nhttps://medium.com/feed/tag/llm",
        height=140,
    )
    rss_urls = [u.strip() for u in rss_text.splitlines() if u.strip()]

    max_items = st.slider("Max items per feed (cap)", 5, 50, 25, 5)

    if st.button("Build / Rebuild Index", use_container_width=True):
        if not api_key:
            st.error("Enter your OpenAI API key.")
        elif not rss_urls:
            st.error("Provide at least one RSS URL.")
        else:
            with st.spinner("Loading RSS + indexing into FAISS..."):
                st.session_state.vs = build_vectorstore_from_medium_rss(
                    rss_urls=rss_urls,
                    openai_api_key=api_key,
                    max_items_per_feed=max_items,
                )
                st.session_state.last_index_time = time.strftime("%Y-%m-%d %H:%M:%S")
            st.success("Index ready.")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vs" not in st.session_state:
    st.info("Build the index from the sidebar to start chatting.")
    st.stop()

st.caption(f"Index last built: {st.session_state.get('last_index_time', 'N/A')}")

# Show chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a concept question (e.g., 'What is LoRA fine-tuning?')")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve
    retriever = st.session_state.vs.as_retriever(search_kwargs={"k": 5})
    retrieved = retriever.get_relevant_documents(prompt)

    # Prepare context for LLM
    context_blocks = []
    for d in retrieved:
        title = (d.metadata or {}).get("title", "Medium Article")
        url = (d.metadata or {}).get("source", "")
        context_blocks.append(f"[{title}]({url})\n\n{d.page_content}")

    context = "\n\n---\n\n".join(context_blocks)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

    system = (
        "You are a helpful teacher. Answer using ONLY the provided context.\n"
        "Explain the concept clearly, then provide a short 'Sources' section.\n"
        "If the context is insufficient, say so and ask for a narrower query.\n"
    )

    full_prompt = f"{system}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{prompt}\n"

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = llm.invoke(full_prompt).content

        st.markdown(answer)

        # Also print a clean, deduped list of sources (explicitly)
        sources = format_sources(retrieved)
        if sources:
            st.markdown("\n\n### Sources (Medium)\n")
            for s in sources:
                line = f"- [{s['title']}]({s['url']})"
                if s.get("published"):
                    line += f" — {s['published']}"
                st.markdown(line)

    st.session_state.messages.append({"role": "assistant", "content": answer})
