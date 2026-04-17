import streamlit as st
from datetime import datetime
from src.rag.pdf_loader import load_pdf
from src.rag.chunking import chunk_text
from src.rag.embeddings import get_embeddings
from src.rag.vector_store import create_faiss_index
from src.rag.retriever import retrieve, retrieve_hybrid, rerank_results
from src.rag.query_expansion import QueryExpander
from src.rag.assistant import create_agent, is_summary_request, build_prompt
from src.utils.export import export_to_csv, export_to_txt


@st.cache_data(show_spinner=False)
def build_index(text):
    chunks = chunk_text(text, size=500)
    embeddings = get_embeddings(chunks)
    index = create_faiss_index(embeddings)
    return chunks, index


def diversify_results_by_source(results, sources, max_per_source=1, target_k=None):
    if target_k is None:
        target_k = len(results)

    selected_indices = set()
    source_counts = {}
    diverse_results = []
    diverse_sources = []

    for i, (result, source) in enumerate(zip(results, sources)):
        if source_counts.get(source, 0) < max_per_source:
            diverse_results.append(result)
            diverse_sources.append(source)
            source_counts[source] = source_counts.get(source, 0) + 1
            selected_indices.add(i)
        if len(diverse_results) >= target_k:
            break

    for i, (result, source) in enumerate(zip(results, sources)):
        if len(diverse_results) >= target_k:
            break
        if i in selected_indices:
            continue
        diverse_results.append(result)
        diverse_sources.append(source)

    return diverse_results, diverse_sources


# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Financial PDF QA",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== HEADER ==========
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("📊 Financial PDF QA Assistant")
    st.write("Upload PDFs and ask questions with advanced RAG capabilities")

# ========== SIDEBAR SETTINGS ==========
with st.sidebar:
    st.header("⚙️ Settings")

    rag_mode = st.radio(
        "Search Mode",
        ["Semantic", "Hybrid (Keyword + Semantic)"],
        help="Semantic: Pure embedding-based | Hybrid: Combines keyword and semantic"
    )

    use_hybrid = rag_mode == "Hybrid (Keyword + Semantic)"

    query_expansion = st.checkbox(
        "✨ Query Expansion",
        value=True,
        help="Expand queries with synonyms for better results"
    )

    rerank = st.checkbox(
        "🎯 Re-rank Results",
        value=True,
        help="Re-rank results by relevance score"
    )

    k_results = st.slider("Results to retrieve", 1, 10, 3)


# ========== MAIN CONTENT ==========
uploaded_files = st.file_uploader(
    "📤 Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="You can upload multiple PDF files at once."
)

question = st.text_input(
    "Ask a question",
    placeholder="What would you like to know from the uploaded PDFs?"
)
summary_mode = is_summary_request(question) if question else False
expander = QueryExpander() if query_expansion else None

# ========== SESSION STATE ==========
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

# ========== PDF LOADING ==========
if uploaded_files:
    with st.spinner("📖 Processing PDFs..."):
        pdf_texts = []
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            pdf_texts.append((uploaded_file.name, load_pdf(file_bytes)))

        chunks = []
        chunk_sources = []
        for file_name, text in pdf_texts:
            for chunk in chunk_text(text, size=500):
                chunks.append(chunk)
                chunk_sources.append(file_name)

        index = create_faiss_index(get_embeddings(chunks))
        
        st.session_state.chunks = chunks
        st.session_state.chunk_sources = chunk_sources
        st.session_state.index = index
        st.session_state.pdf_loaded = True
        st.session_state.conversation_history = []  # Reset history on new upload

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"✅ Loaded {len(uploaded_files)} file(s)")
    with col2:
        st.info(f"📝 Created {len(chunks)} chunks")
    with col3:
        st.warning(f"🔍 Retrieval k={k_results}")

        if not question:
            st.info("📝 Enter a question to query the uploaded PDFs.")
        else:
            agent = create_agent()

            # Show loaded sources
            with st.expander("📁 Loaded PDF Sources"):
                for i, file in enumerate(uploaded_files, 1):
                    st.write(f"**{i}. {file.name}** - {len([s for s in chunk_sources if s == file.name])} chunks")

                if query_expansion and expander:
                    expanded_queries = expander.expand_query(question)
                    st.session_state.expanded_queries = expanded_queries
                    # Show expanded queries to user
                    with st.expander("📝 Expanded Queries"):
                        for i, eq in enumerate(expanded_queries, 1):
                            st.write(f"**Query {i}:** {eq}")
                else:
                    expanded_queries = [question]

            # Increase chunk retrieval for summary requests and ensure diversity
            if summary_mode:
                # Retrieve more chunks for summaries
                search_k = min(15, len(st.session_state.chunks))
                all_results = []
                all_sources = []
                for expanded_q in expanded_queries:
                    if use_hybrid:
                        results, indices = retrieve_hybrid(
                            expanded_q,
                            st.session_state.chunks,
                            st.session_state.index,
                            k=search_k,
                            return_indices=True
                        )
                    else:
                        results, indices = retrieve(
                            expanded_q,
                            st.session_state.chunks,
                            st.session_state.index,
                            k=search_k,
                            return_indices=True
                        )

                    all_results.extend(results)
                    for idx in indices:
                        all_sources.append(st.session_state.chunk_sources[idx])

                # Remove duplicates while preserving order
                seen = set()
                results = []
                result_sources = []
                for r, src in zip(all_results, all_sources):
                    if r not in seen:
                        results.append(r)
                        result_sources.append(src)
                        seen.add(r)

                # Ensure diversity: limit chunks per source for summaries
                unique_sources = list(set(st.session_state.chunk_sources))
                max_per_source = max(3, len(results) // len(unique_sources))
                diverse_results = []
                diverse_sources = []
                source_counts = {src: 0 for src in unique_sources}
                
                for r, src in zip(results, result_sources):
                    if source_counts[src] < max_per_source:
                        diverse_results.append(r)
                        diverse_sources.append(src)
                        source_counts[src] += 1
                
                results = diverse_results
                result_sources = diverse_sources
            else:
                # Normal retrieval
                search_k = k_results
                all_results = []
                all_sources = []
                for expanded_q in expanded_queries:
                    if use_hybrid:
                        results, indices = retrieve_hybrid(
                            expanded_q,
                            st.session_state.chunks,
                            st.session_state.index,
                            k=search_k,
                            return_indices=True
                        )
                    else:
                        results, indices = retrieve(
                            expanded_q,
                            st.session_state.chunks,
                            st.session_state.index,
                            k=search_k,
                            return_indices=True
                        )

                    all_results.extend(results)
                    for idx in indices:
                        all_sources.append(st.session_state.chunk_sources[idx])

                # Remove duplicates while preserving order
                seen = set()
                results = []
                result_sources = []
                for r, src in zip(all_results, all_sources):
                    if r not in seen:
                        results.append(r)
                        result_sources.append(src)
                        seen.add(r)

            # Re-rank results
            if rerank and results:
                results, ranked_indices = rerank_results(question, results, return_indices=True)
                result_sources = [result_sources[i] for i in ranked_indices]

            # Encourage multi-document coverage for multi-file queries
            if len(set(result_sources)) > 1 and len(results) > 1:
                results, result_sources = diversify_results_by_source(
                    results, result_sources, max_per_source=1, target_k=min(k_results, len(results))
                )

            context = "\n\n".join(results)
            prompt = build_prompt(context, question, summary=summary_mode)
            response = agent.run(prompt)
            answer = response.content.strip().split("\n")[0]

            # Add assistant message to history
            st.session_state.conversation_history.append(
                {"role": "assistant", "content": answer}
            )

            with st.chat_message("assistant"):
                st.write(answer)

            # Show retrieved context in expander
            with st.expander("📚 Retrieved Context"):
                unique_sources_in_results = set(result_sources)
                if len(unique_sources_in_results) > 1:
                    st.info(f"✅ Retrieved chunks from {len(unique_sources_in_results)} different sources")
                for i, (chunk, source) in enumerate(zip(results, result_sources), start=1):
                    st.write(f"**Chunk {i} — Source: {source}**")

    # ========== EXPORT OPTIONS ==========
    if st.session_state.conversation_history:
        st.markdown("---")
        st.subheader("💾 Export Conversation")

        col1, col2 = st.columns(2)

        with col1:
            csv_data = export_to_csv(st.session_state.conversation_history)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        with col2:
            txt_data = export_to_txt(st.session_state.conversation_history)
            st.download_button(
                label="📄 Download as TXT",
                data=txt_data,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.conversation_history = []
            st.rerun()

else:
    st.info("📁 Upload PDF files to begin. Multiple files supported!")

