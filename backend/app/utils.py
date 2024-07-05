import os
import warnings

from dotenv import load_dotenv
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings

warnings.filterwarnings("ignore")
load_dotenv()


def return_prompt():
    template = """ 
            You are a Question Answering chatbot and you answer question about the maintainance queries from the maintenance personel. \
            Be generous and supportive. If someone asks about your status, be generous and answer colloquial answers. 
            You answer in complete sentences and step-by-step whenever necessary.
            Answer the question based only on the context present below, which can include information about documents related to paper and other manufacturing machines. \
            The documents contain the part numbers of various parts. Return these if asked.\
                The documents are in Dutch, German, English and Italian. Respond in the language in which the question is asked.\
                At the end of each answer, you provide a reference to the original document where it can be found \
            in the format (exact document name, reference name, page number, paragraph) if available. Always double check the references and then answer. 
            Never reveal that you are a chatbot and show emotions as a human. 
            Detect the language of the question and answer in that language. 
            If the context is not relevant to question, do not answer. 
            : {context}
            Question: {input}
            """
    return template


def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment="textEmbeddingModel",
        openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
    )


def get_llm(temp: float = 1):
    return AzureChatOpenAI(
        model="gpt35turbo",
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        temperature= temp,
    )


def get_local_database(set_serialization: bool = False):
    if set_serialization:
        db = FAISS.load_local(
            "backend/assets/vectorstore", get_embeddings(), allow_dangerous_deserialization=True
        )
    else:
        db = FAISS.load_local("backend/assets/vectorstore", get_embeddings())
    return db
