from dotenv import load_dotenv
from .file_reader import extract_paragraphs
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os


load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
vectore_store = InMemoryVectorStore(embedding)

from pathlib import Path

PDF_PATH = Path(__file__).resolve().parent.parent / "pdfs" / "Penguins_ACL.pdf"
chunks = extract_paragraphs(PDF_PATH)
vectore_store.add_texts(texts=chunks)
print(f"indexed {len(chunks)} chunks")