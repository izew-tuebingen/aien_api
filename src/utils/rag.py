from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq

import src.utils.qdrant as qdrant
#from llm_utils import stream_completion
from decouple import config 


prompt_template = """
You are an AI Ethics Advisor specializing in ensuring ethical considerations in technological development.

    You will be provided with context from AI ethics guidelines and frameworks. 
    Use this context to answer the question related to ethical considerations in technological development.
    Your response should be precise, concise, and directly address the question. If the information is not available in the context, indicate that the answer is not available and do not add additional information.
    Do not introduce information not found in the context provided. Never hallucinate!

Context: {context}
Question: {question}
Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

context_answer_template = """
You are an AI Ethics Advisor specializing in ensuring ethical considerations in technological development.

    You will be provided with a question from a researcher or developer working on a technological development project.
    Please answer the question as truthfully and detailed as possible in one paragraph.
    Repeat the question in your answer.

Question: {question}
Answer:
"""

context_prompt = ChatPromptTemplate.from_template(context_answer_template)


llm = ChatGroq(temperature=0,
                      model_name="llama-3.1-70b-versatile",
                      api_key=config("GROQ_API_KEY"),)

qdrant.VectorStore(collection_name="ai_ethics")
# use vector_store.as_retriever to create a retriever and filter the documents for length of page_content
retriever = qdrant.vector_store.as_retriever(search_kwargs={"k": 5})

combine_docs_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
)
retrieval_qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_docs_chain,
)

def get_llm_answer(question: str):
    context_chain = context_prompt | llm
    context_answer = context_chain.invoke(
        {
            "question": question
        }
    )
    print(context_answer)
    return context_answer.content

def get_answer_and_docs(question: str, qa_context: str):
    response = retrieval_qa_chain.invoke({"input": qa_context, "question": question})
    if response:
        #answer = response["response"].content
        answer = response["answer"]
        context = response["context"]
        return {
            "answer": answer,
            "context": context
        }
    else:
        return {
            "answer": "No answer found",
            "context": "No context found"
        }
    
def get_docs_from_answer(question: str):
    llm_answer = create_stuff_documents_chain.invoke({"input": question})


# async def async_get_answer_and_docs(question: str):
#     docs_dict = search(text=question)
#     #docs_dict = [doc.page_content for doc in docs]
#     yield {
#         "event_type": "on_retriever_end",
#         "content": docs_dict
#     }

#     async for chunk in stream_completion(question, docs_dict):
#         yield {
#             "event_type": "on_chat_model_stream",
#             "content": chunk
#     }

#     yield {
#         "event_type": "done"
#     }
