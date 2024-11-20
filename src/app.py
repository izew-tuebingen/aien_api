from modal import Image, App, asgi_app, Secret
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from decouple import config

from src.utils.rag import get_answer_and_docs, get_llm_answer

app = App("aien_api")
app.image = Image.debian_slim().poetry_install_from_file("./pyproject.toml")

auth_scheme = HTTPBearer()

@app.function(secrets=[Secret.from_dotenv()])
@asgi_app()
def endpoint():
    app = FastAPI(title="AI Ethics Navigator API", description="API for the AI Ethics Navigator", version="0.1")

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
        context_answer = get_llm_answer(question=q.question)
        response = get_answer_and_docs(question=q.question, qa_context=context_answer)
        response_dict = {
            "question": q.question,
            "answer": response["answer"],
            "documents": [doc.dict() for doc in response["context"]]
        }
        return JSONResponse(content=response_dict, status_code=200)
    
    return app


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="localhost", port=8000)