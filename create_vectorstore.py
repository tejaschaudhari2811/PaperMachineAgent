from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

all_documents = []

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        pdf_pages = PyPDFLoader(os.path.join("data",file)).load()
        all_documents.extend(pdf_pages)
    if file.endswith(".txt"):
        text_pages = TextLoader(os.path.join("data",file)).load()
        all_documents.extend(text_pages)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment="textEmbeddingModel",
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
)

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

splits = text_splitter.split_documents(all_documents)

db = FAISS.from_documents(splits, embeddings)

db.save_local("vectorstore")