import os
import modal
from src.app import app, volume, image

# This script runs on Modal to leverage cloud vCPUs for fast embedding
@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="L4", # Accelerate embedding with an NVIDIA L4 GPU
    timeout=3600 # 1 hour timeout
)
def ingest_data():
    import src.utils.parse as parse
    import src.utils.vector_store as vs
    import json
    
    import pickle
    
    # 1. Load documents from the storage (with Pickle Caching)
    doc_dir = "/data/documents"
    metadata_path = "/data/registered_documents.json"
    pkl_path = "/data/parsed_documents.pkl"
    
    if os.path.exists(pkl_path):
        print(f"Loading pre-parsed chunks from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            docs = pickle.load(f)
    else:
        if not os.path.exists(doc_dir):
            print(f"Error: {doc_dir} not found. Did you upload it?")
            return
            
        print("Parsing documents...")
        docs = parse.parse_and_split_downloaded_documents(
            dirpath=doc_dir, 
            metadata_path=metadata_path
        )
        print(f"Saving {len(docs)} parsed chunks to cache...")
        with open(pkl_path, "wb") as f:
            pickle.dump(docs, f)
    
    print(f"Total chunks to embed: {len(docs)}")
    
    # 2. Set GPU flag for the vector store
    os.environ["MODAL_GPU"] = "True"
    
    import shutil
    # 3. Embed and upload to LanceDB in a local POSIX-compliant workspace
    local_workspace = "/tmp/lancedb_workspace"
    os.environ["VECTOR_DB_PATH"] = local_workspace
    
    store = vs.init_store()
    # Use larger batch size for GPU performance
    store.embed_and_upload_documents(docs, batch_size=128, replace=True)
    
    print("Moving finalized database to the Modal Volume...")
    target_dir = "/data/lancedb_vectors"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)
        
    shutil.copytree(local_workspace, target_dir, dirs_exist_ok=True)
    
    # CRITICAL: Commit the volume changes to permanent cloud storage
    volume.commit()
    
    print("Ingestion complete and changes committed to volume!")

if __name__ == "__main__":
    with modal.Retrying().run():
        ingest_data.remote()
