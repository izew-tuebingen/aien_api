from modal import Image, App, asgi_app, Secret
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from decouple import config

import os
import sys

# Use official NVIDIA CUDA image to guarantee onnxruntime-gpu driver compatibility
image = (Image
             .from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12")
             .poetry_install_from_file("./pyproject.toml")
             .pip_install("onnxruntime-gpu")
             .add_local_python_source("src")
)

# Persistent high-performance storage for LanceDB & Documents
import modal
volume = modal.Volume.from_name("aien-vector-store", create_if_missing=True)

app = App(
    name="ai-guidelines-guru",
    image=image,
    secrets=[Secret.from_dotenv()],
    volumes={"/data": volume},
)

# Set environment variable for the local store
os.environ["VECTOR_DB_PATH"] = "/data/lancedb_vectors"
os.environ["FLASHRANK_CACHE"] = "/data/flashrank"

auth_scheme = HTTPBearer()

from src.utils.rag import get_answer_and_docs, stream_answer_and_docs

@app.function(image=image)
def debug_paths():
    import sys
    import os
    print("Current working directory:", os.getcwd())
    print("Python path:", sys.path)
    print("Files in current dir:", os.listdir("."))
    if os.path.exists("src"):
        print("Files in src:", os.listdir("src"))
    return "Debug complete"

@app.function(secrets=[Secret.from_dotenv()])
@asgi_app()
def endpoint():
    app = FastAPI(title="AI Guidelines Guru", description="Expert navigation of AI ethics and guidelines", version="1.0")

    #Add CORS middleware
    origins = [
        "https://ai-ethics-navigator.streamlit.app",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
    )

    class Question(BaseModel):
        question: str

    @app.post("/api/qa", description="Ask a question to the AI Ethics Navigator")
    def qa(q: Question, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        if token.credentials != config("AIEN_AUTH_TOKEN"):
            return JSONResponse(content={"error": "Incorrect bearer token"}, status_code=401)
        # avoid crash looping if the methods are not reachable
        if not hasattr(get_answer_and_docs, '__call__'):
            return JSONResponse(content={"error": "Service not available"}, status_code=503)
        response = get_answer_and_docs(question=q.question)
        response_dict = {
            "question": q.question,
            "answer": response["answer"],
            "documents": [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in response["context"]
            ],
        }
        return JSONResponse(content=response_dict, status_code=200)

    @app.post("/api/qa/stream", description="Stream an answer from the AI Ethics Navigator")
    async def qa_stream(q: Question, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        if token.credentials != config("AIEN_AUTH_TOKEN"):
            return JSONResponse(content={"error": "Incorrect bearer token"}, status_code=401)

        import json

        async def event_generator():
            async for event in stream_answer_and_docs(question=q.question):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="localhost", port=8000)