from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a text about it"}
  ]
)

print(completion.choices[0].message);

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create and save FAISS vector store from text chunks."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Set up the conversational chain."""
    prompt_template = """
    Use the context below to answer the question as thoroughly as possible.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model=openai.ChatCompletion, chain_type="stuff", prompt=prompt)

@app.post("/ask_question")
async def ask_question(question: str = Form(...)):
    """Endpoint to ask a question."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local("faiss_index", embeddings)
    docs = vector_store.similarity_search(question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return JSONResponse(status_code=200, content={"response": response["output_text"]})
