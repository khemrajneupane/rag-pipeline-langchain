from fastapi import FastAPI, UploadFile, File, Header, HTTPException,Depends,Request
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from fastapi.responses import PlainTextResponse
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from dotenv import load_dotenv
from pydantic import BaseModel
from pinecone import Pinecone
from slowapi import Limiter
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")
api_key_protection = os.getenv("API_KEY_PROTECTION")

# ---------------------------------------------------------
# YOUR EXISTING FUNCTIONS (UNCHANGED)
# ---------------------------------------------------------
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

def chunk_data(docs, chunk_size=800, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(docs)
    return docs

# ---------------------------------------------------------
# EMBEDDING + PINECONE + VECTOR STORE
# ---------------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

pc = Pinecone(api_key=pinecone_api_key)
index_name = pinecone_index
index = pc.Index(index_name)

# Initialize vector store, pointing to existing pinecone index. (This is not adding embedding to db)
# Makes vector_store object available globally.
# We can call as_retriever() on it even before any documents are uploaded.

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# ---------------------------------------------------------
# SETUP RETRIEVER initially to have a it available globally for empty or already existing old data in Pinecone)
# ---------------------------------------------------------

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Increased k to 5 for better retrieval
)

llm = ChatOpenAI(model="gpt-4o")

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

prompt = PromptTemplate.from_template("""
You are a helpful assistant that answers questions based ONLY on the provided context.

Context:
{context}

Question: {query}

Answer the question based solely on the context above. If the context doesn't contain enough information to answer the question fully, say "I could not find the answer in the documents." but try to provide whatever relevant information you can find.

Answer:
""")
# RAG pipeline creation. LangChain requires a function that receives the whole input object (x)
rag_chain = (
    RunnableMap({
        "query": lambda x: x["query"], #simply picks the question
        "context": lambda x: format_docs(retriever.invoke(x["query"])) # retrieves relevant text from Pinecone
    })
    | prompt
    | llm
    | StrOutputParser() #Extracts the plain text answer from the LLM response.
)

# ---------------------------------------------------------
# FASTAPI SERVER
# ---------------------------------------------------------
app = FastAPI(title="RAG Q/A API")

# ---------------------------------------------------------
# Verify API key function to protect routes
# ---------------------------------------------------------
async def verify_api_key(api_key: str = Header(None, alias="X-API-Key")):
    if api_key != api_key_protection:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

# ---------------------------------------------------------
# Rate-limit handler
# ---------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return PlainTextResponse("Too Many Requests", status_code=429)

# Allow frontend calls (Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set specific domain later
    allow_methods=["*"],
    allow_headers=["*"]
)

# Input model
class Question(BaseModel):
    query: str


# ---------------------------------------------------------
# API ENDPOINT /ask
# ---------------------------------------------------------
@app.post("/ask")
@limiter.limit("1/minute")  # each IP: max 5 questions per minute. Can be adjusted 10/hour,50/day,1/sec etc.
async def ask_question(
    request: Request, 
    payload: Question,
    api_key: str = Depends(verify_api_key)
    ):
    answer = rag_chain.invoke({"query": payload.query})
    return {"answer": answer}

# ---------------------------------------------------------
# API ENDPOINT — UPLOAD NEW PDF + DELETE OLD FILES 
# ---------------------------------------------------------
@app.post("/upload_pdf")
@limiter.limit("1/minute") 
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
    ):
    global vector_store, retriever, rag_chain

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    folder = "documents"

    # Make sure folder exists
    os.makedirs(folder, exist_ok=True)

    # Remove existing PDFs
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path):
            os.remove(path)

    # Save new PDF
    pdf_path = os.path.join(folder, file.filename)
    with open(pdf_path, "wb") as buffer:
        buffer.write(file.file.read())

    try:
        # Rebuild documents and chunks
        new_docs = read_doc(folder)
        if len(new_docs) == 0:
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        chunked = chunk_data(new_docs)

        # Clear Pinecone before rebuilding. Delete existing vectors
        try:
            index.delete(delete_all=True)
        except Exception as e:
            print(f"Note during delete: {e}")

        # Recreate vector store with new documents
        # This upserts (adds) the vectors into the Pinecone index. This is required here because this is a new document being uploaded.
        vector_store = PineconeVectorStore.from_documents(
            documents=chunked,
            embedding=embeddings,
            index_name=index_name
        )

        # Recreate retriever and rag_chain
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        rag_chain = (
            RunnableMap({
                "query": lambda x: x["query"],
                "context": lambda x: format_docs(retriever.invoke(x["query"]))
            })
            | prompt
            | llm
            | StrOutputParser()
        )

    except Exception as e:
        # Clean up on error
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# ---------------------------------------------------------
# API ENDPOINT —DELETE OLD FILES
# ---------------------------------------------------------
@app.delete("/delete_index")
async def delete_index():
    try:
        index.delete(delete_all=True)
        return {"message": "Index deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting index: {str(e)}")

# ---------------------------------------------------------
# RUN SERVER:
# uvicorn app:app --reload
# ---------------------------------------------------------