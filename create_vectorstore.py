from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

loader = PyPDFLoader(
    "data/andritz-primeline-tissue-machines-data.pdf"
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment="textEmbeddingModel",
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
)

pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

splits = text_splitter.split_documents(pages)

db = FAISS.from_documents(splits, embeddings)

db.save_local("vectorstore")