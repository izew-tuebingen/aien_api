from langchain_core.documents import Document
from src.utils.rag import format_context

def test_format_context():
    # Case 1: Document with a valid URL
    doc1 = Document(
        page_content="Content of document 1",
        metadata={
            "document_name": "Test Document 1",
            "page": 5,
            "document_url": "https://example.com/doc1"
        }
    )
    # Case 2: Document with a 'link' key instead of 'document_url'
    doc2 = Document(
        page_content="Content of document 2",
        metadata={
            "document_name": "Test Document 2",
            "page": "N/A",
            "link": "https://example.com/doc2"
        }
    )
    # Case 3: Document with no link or 'N/A' link
    doc3 = Document(
        page_content="Content of document 3",
        metadata={
            "document_name": "Test Document 3",
            "page": 10,
            "document_url": "N/A"
        }
    )

    formatted = format_context([doc1, doc2, doc3])
    print("Formatted output:")
    print(formatted)

if __name__ == "__main__":
    test_format_context()
