import os
import warnings
import streamlit as st
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

warnings.filterwarnings("ignore")
load_dotenv()

# Create streamlit page
st.title("Paper Machine Agent: \n Talk to manual for Andritz Tissue Machine")
st.image("logos/logo_insights.png")
st.sidebar.image("logos/book.jpg")
st.sidebar.link_button("Go to the Manual", os.getenv("BOOK_LINK"))

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment="textEmbeddingModel",
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
)

db = FAISS.load_local("vectorstore", embeddings)
retriever = db.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = AzureChatOpenAI(
    model="gpt35turbo",
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Prompt template
template = """ You are a Question Answering bot. You answer in complete sentences and step-by-step whenever necessary.
You always provide references from the context report which is a report about a Tissue Machine from Andtritz with page number and paragraph start sentence in 
double quotation marks. 
Answer the question based only on the following context, which can include information about Data:
{context}
Question: {question}
"""

prompt2 = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt2
    | llm
    | StrOutputParser()
)

st.markdown(
    """Enter your text (prompt) in the following box. Always prompt as concrete as possible and \
             always cross check the references in the manual that is provided. Please provide Feedback to Tejas :blush:."""
)

user_message_1 = st.text_input(label="Please enter your question...")

if user_message_1:
    response = rag_chain.invoke(user_message_1)

    st.markdown(response)
