from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from flashrank import Ranker, RerankRequest

import src.utils.vector_store as vs
from decouple import config
import os


# --- Prompt with citation and confidence instructions ---

prompt_template = """You are the AI Guidelines Guru, a specialized expert in navigating ethical frameworks and guidelines for technological development.

You will be provided with context snippets from AI ethics guidelines and frameworks, each labeled with a source document name and page number.

Instructions:
- Use ONLY the provided context to answer the question. Do not introduce external knowledge.
- Be precise and concise. Directly address the question.
- **Cite your sources** by referencing the document name and page number in parentheses, e.g. (Source: "EU AI Act Guidelines", p. 12). Only display page numbers if they are available (do not display "N/A" as a page number). 
- At the end of your answer, include a confidence statement: "Confidence: [high/medium/low]" based on how well the provided context covers the question.
- If the context does not contain enough information, say: "The provided documents do not contain sufficient information to answer this question." and set confidence to low.

Context:
{context}

Question: {input}
Answer:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# --- LLM ---

llm = ChatGroq(
    temperature=0,
    model_name="openai/gpt-oss-120b",
    api_key=config("GROQ_API_KEY"),
)

# --- Vector store (local LanceDB) ---

def get_vector_store():
    """Lazy-load the vector store to prevent local import-time execution on Modal."""
    if vs.vector_store is None:
        vs.init_store()
    return vs.vector_store

# --- Cross-encoder re-ranker ---

_reranker = None

def get_reranker():
    """Lazy-load the reranker model to avoid heavy operations on local import."""
    global _reranker
    if _reranker is None:
        cache_dir = os.environ.get("FLASHRANK_CACHE", "/tmp/flashrank")
        _reranker = Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir=cache_dir)
    return _reranker


def rerank_documents(query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
    """Re-rank retrieved documents using a cross-encoder for better precision."""
    if not documents:
        return []

    # Build rerank request
    passages = [
        {"id": i, "text": doc.page_content, "meta": doc.metadata}
        for i, doc in enumerate(documents)
    ]
    rerank_request = RerankRequest(query=query, passages=passages)
    results = get_reranker().rerank(rerank_request)

    # Map back to LangChain Documents, preserving metadata
    reranked_docs = []
    for result in results[:top_n]:
        idx = result["id"]
        original_doc = documents[idx]
        reranked_docs.append(original_doc)

    return reranked_docs


def format_context(documents: list[Document]) -> str:
    """Format documents into a numbered context string with source metadata."""
    formatted = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("document_name", "Unknown")
        page = doc.metadata.get("page")
        
        # Clean page number
        if page is not None:
            page_str = str(page).strip()
            if page_str.lower() in ("n/a", "null", "—", "-", ""):
                page_str = None
        else:
            page_str = None
            
        link = doc.metadata.get("document_url") or doc.metadata.get("link") or "N/A"
        if link and link != "N/A":
            source_str = f'<a href="{link}">"{source}"</a>'
        else:
            source_str = f'"{source}"'
            
        if page_str:
            formatted.append(
                f"[{i}] Source: {source_str}, Page: {page_str}\n{doc.page_content}"
            )
        else:
            formatted.append(
                f"[{i}] Source: {source_str}\n{doc.page_content}"
            )
    return "\n\n".join(formatted)


def get_answer_and_docs(question: str) -> dict:
    """Retrieve, re-rank, and generate an answer with citations."""
    vector_store = get_vector_store()
    # Step 1: Retrieve top-20 candidates from local LanceDB
    candidates = vector_store.similarity_search(question, k=25)

    # Filter out structurally irrelevant PDF artifacts (e.g., Table of Contents headers)
    # Junk chunks are typically very short but get artificially high cross-encoder scores 
    # due to extreme keyword density (e.g., "What are the Ethics of AI? 1 12").
    valid_candidates = [doc for doc in candidates if len(doc.page_content) > 100]
    if not valid_candidates:
        valid_candidates = candidates  # Fallback

    if not valid_candidates:
        return {
            "answer": "No relevant documents found in the knowledge base.",
            "context": [],
        }

    # Step 2: Re-rank to top-5 using cross-encoder
    top_docs = rerank_documents(question, valid_candidates, top_n=5)

    # Step 3: Format context and generate answer via LLM
    context_str = format_context(top_docs)
    chain = prompt | llm
    response = chain.invoke({"context": context_str, "input": question})

    return {
        "answer": response.content,
        "context": top_docs,
    }


async def stream_answer_and_docs(question: str):
    """Stream the answer token-by-token, yielding documents first."""
    vector_store = get_vector_store()
    # Step 1: Retrieve & re-rank
    candidates = vector_store.similarity_search(question, k=25)
    
    # Filter out short junk chunks (PDF artifacts) before re-ranking
    valid_candidates = [doc for doc in candidates if len(doc.page_content) > 100]
    if not valid_candidates:
        valid_candidates = candidates
        
    top_docs = rerank_documents(question, valid_candidates, top_n=5)

    # Yield retrieved documents first
    yield {
        "event_type": "on_retriever_end",
        "content": [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in top_docs
        ],
    }

    # Step 2: Stream LLM response
    context_str = format_context(top_docs)
    chain = prompt | llm

    async for chunk in chain.astream({"context": context_str, "input": question}):
        yield {
            "event_type": "on_chat_model_stream",
            "content": chunk.content,
        }

    yield {"event_type": "done"}
