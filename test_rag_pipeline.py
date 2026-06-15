import modal
import os
from src.app import app, volume

@app.function(volumes={"/data": volume})
def test():
    from src.utils.rag import get_answer_and_docs, rerank_documents, get_vector_store
    
    question = "What are the core principles of AI transparency?"
    print(f"Question: {question}")
    
    vector_store = get_vector_store()
    dense_docs = vector_store.similarity_search(question, k=10)
    reranked_docs = rerank_documents(question, dense_docs, top_n=5)
    
    print("\n\n=== FULL TEXT OF SOURCE 1 ===")
    print(f"Document: {reranked_docs[0].metadata.get('document_name')}")
    print(reranked_docs[0].page_content)
    print("===============================\n\n")

if __name__ == "__main__":
    with modal.Retrying().run():
        test.remote()
