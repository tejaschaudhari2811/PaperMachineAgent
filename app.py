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
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
warnings.filterwarnings("ignore")
load_dotenv()

# Create streamlit page
st.title("Chat with maintenance documents.")
st.text("Brought to you by:")
st.image("logos/logo_insights.png")

with st.sidebar:
    st.write(
    """The documents can be found in the following sharepoint folder. """)
    st.link_button(label="documents",url="https://wepaeu.sharepoint.com/:f:/s/InsightsWEPA.digital-WEPA.digitalinternal/Er5bvpbhCZRMuah_7IR9LW4BxTCc__k_5pwJI1yDsoJ1_w?e=BdBCjm")
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment="textEmbeddingModel",
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)
user = st.chat_input("Please Enter your question..")
st.markdown("You can also chat with your pdf. Please upload the pdf here")
pdf = st.file_uploader("Upload the file", type="pdf")
pdf_db = None
if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20,length_function=len,
    is_separator_regex=False,)
    chunks = text_splitter.split_text(text)
    pdf_db = FAISS.from_texts(chunks, embedding=embeddings)

if pdf_db is not None:
    db.merge_from(pdf_db)
    retriever = db.as_retriever()
else:
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
template = """ 
You are a Question Answering chatbot and you answer question about the maintainance queries from the maintenance personel. \
Be generous and supportive. If someone asks about your status, be generous and answer colloquial answers. 
You answer in complete sentences and step-by-step whenever necessary.
Answer the question based only on the context present below, which can include information about documents related to paper and other manufacturing machines. \
The documents contain the part numbers of various parts. Return these if asked.\
    The documents are in Dutch, German, English and Italian. Respond in the language in which the question is asked.\
    At the end of each answer, you provide a reference to the original document where it can be found \
in the format (document name, reference name, page number, paragraph) whichever applicable. Always double check the references and then answer. 
Never reveal that you are a chatbot and show emotions as a human. 
If the context is not relevant to question, do not answer. 
: {context}
Question: {question}
"""

prompt2 = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt2
    | llm
    | StrOutputParser()
)


def get_response(query):
    return rag_chain.invoke(query)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("User"):
            st.markdown(message.content)
    else:
        with st.chat_message("WEPA Bot"):
            st.markdown(message.content)


if user is not None and user != "":
    st.session_state.chat_history.append(HumanMessage(user))
    with st.chat_message("User"):
        st.markdown(user)

    with st.chat_message("WEPA Bot"):
        response = get_response(user)
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(response))
