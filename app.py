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
st.title("Chat with your Documents")
st.text("""1. The Adritz PrimeLine Tissue Machines \n2. Laptop Manual Lenovo Thinkpad \n3. Environment, Health and Safety Regulations for Paper Mills.""")
st.text("Brought to you by:")
st.image("logos/logo_insights.png")
with st.sidebar:
    st.image("logos/book.jpg", width=200)
    st.link_button("Go to the Manual", os.getenv("BOOK_LINK"))
    st.image("logos/laptop.png")
    st.link_button("Go to the Manual", "https://download.lenovo.com/pccbbs/mobiles_pdf/x250_hmm_en_sp40f30022.pdf")
    st.image("logos/hands.png")
    st.link_button("Go to the Manual", "https://documents1.worldbank.org/curated/en/205611489661890765/text/113557-WP-ENGLISH-Pulp-and-Paper-Mills-PUBLIC.txt")

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment="textEmbeddingModel",
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
)

db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = AzureChatOpenAI(
    model="gpt35turbo",
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Prompt template
template = """ You are a Question Answering bot. You answer in complete sentences and step-by-step whenever necessary.
Answer the question based only on the context present below, which can include information about the safety regulations, or \
machine or laptop maintenance information. At the end of each answer, you provide a reference to the original document where it can be found \
     in the format (reference name, page number, paragraph). Always double check the references and then answer.
     If the context is not relevant to question, do not answer. :
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
    """Enter your Question (prompt) in the following box. Always ask a question as relevant to the documents as possible and \
             always cross check the references in the manual that is provided. Please provide Feedback to Tejas :blush:."""
)

user_message_1 = st.text_input(label="Please enter your question...")

if user_message_1:
    response = rag_chain.invoke(user_message_1)

    st.markdown(response)
