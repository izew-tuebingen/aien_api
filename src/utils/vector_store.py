import lancedb
import os
import pandas as pd
from tqdm import tqdm
from fastembed import TextEmbedding
from langchain_core.documents import Document

# --- Configuration ---
TABLE_NAME = "aien_ethics"

class LocalVectorStore:
    def __init__(self, model_name="intfloat/multilingual-e5-large"):
        # Initialize embedding model (Advanced Large Model)
        # Use GPU only if MODAL_GPU is set (to avoid errors on Mac)
        providers = ["CPUExecutionProvider"]
        if os.getenv("MODAL_GPU") == "True":
            providers = ["CUDAExecutionProvider"] + providers
            
        self.encoder = TextEmbedding(
            model_name=model_name,
            providers=providers
        )
        
        # Connect to LanceDB (local file storage)
        self.db_path = os.getenv("VECTOR_DB_PATH", "lancedb_vectors")
        os.makedirs(self.db_path, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        self.table = None
        self._load_table()

    def _load_table(self):
        """Open the table if it exists."""
        if TABLE_NAME in self.db.table_names():
            self.table = self.db.open_table(TABLE_NAME)

    def delete_collection(self):
        """Delete the local table."""
        if TABLE_NAME in self.db.table_names():
            self.db.drop_table(TABLE_NAME)
            self.table = None
            print(f"Table {TABLE_NAME} deleted.")

    def embed_and_upload_documents(self, documents: list[Document], batch_size: int = 32, replace: bool = False):
        """Embed and upload documents to LanceDB with progress tracking."""
        data = []
        texts = [doc.page_content for doc in documents]
        
        print(f"Embedding {len(documents)} chunks...")
        embeddings = []
        
        # FastEmbed.embed returns a generator. Using tqdm to show progress.
        for batch in tqdm(self.encoder.embed(texts, batch_size=batch_size), total=len(texts), desc="Ingesting"):
            embeddings.append(batch)
        
        for i, doc in enumerate(documents):
            entry = {
                "vector": embeddings[i],
                "text": doc.page_content,
            }
            # Flatten metadata into the entry and enforce type consistency
            for k, v in doc.metadata.items():
                # PyArrow requires strict schema types. PDF metadata often mixes strings and floats (e.g., page numbers).
                # Force everything to string to prevent 'Expected bytes, got float' errors.
                if v is None:
                    entry[k] = ""
                elif isinstance(v, (list, dict)):
                    import json
                    entry[k] = json.dumps(v)
                else:
                    entry[k] = str(v)
            data.append(entry)
            
        print(f"Adding {len(data)} chunks to LanceDB table '{TABLE_NAME}'...")
        # Create or update the table
        if replace or TABLE_NAME not in self.db.table_names():
            print(f"Creating LanceDB table '{TABLE_NAME}'...")
            self.table = self.db.create_table(TABLE_NAME, data=data, mode="overwrite")
        else:
            self.table = self.db.open_table(TABLE_NAME)
            self.table.add(data)
            
        # Ensure Full-Text Search index is updated
        self.table.create_fts_index("text", replace=True)
        print(f"Successfully indexed {len(data)} documents.")

    def similarity_search(self, query: str, k: int = 20) -> list[Document]:
        """Dense Vector search followed by downstream FlashRank."""
        if not self.table:
            return []
            
        # 1. Embed the user string into a 1024-dim dense vector using FastEmbed
        query_vector = list(self.encoder.embed([query]))[0]
            
        # 2. Search LanceDB using the explicit numeric float array (ANN)
        results = (
            self.table.search(query_vector)
            .limit(k)
            .to_pandas()
        )
        
        docs = []
        for _, row in results.iterrows():
            metadata = row.to_dict()
            page_content = metadata.pop("text")
            metadata.pop("vector") if "vector" in metadata else None
            metadata.pop("_distance") if "_distance" in metadata else None
            metadata.pop("_score") if "_score" in metadata else None
            
            docs.append(Document(page_content=page_content, metadata=metadata))
            
        return docs

# Global instance
vector_store = None

def init_store():
    global vector_store
    if vector_store is None:
        vector_store = LocalVectorStore()
    return vector_store
