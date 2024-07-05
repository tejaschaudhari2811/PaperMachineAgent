import streamlit as st
import pandas as pd
import uuid
import sys
from pathlib import Path
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import get_llm, get_local_database, return_prompt
from pathlib import Path
import sys

# Get the current script's directory
current_path = Path(__file__).resolve().parent.parent

# Append 'backend' directory to sys.path
sys.path.append(str(current_path))

# Now you should be able to import modules from the backend folder
from database.azure_cosmos_module import get_client

# Set up connection to Azure Cosmos DB
client = get_client()

db = client.get_database("BoardEfficiencyCompanion")

chat_history = db.get_collection("ChatHistory")

# Create streamlit page
st.title("Chat with maintenance documents.")
st.write("Brought to you by:")
st.image("logos/logo_insights.png")

with st.sidebar:
    st.write(
    """The documents can be found in the following sharepoint folder. """)
    st.link_button(label="documents",url="https://wepaeu.sharepoint.com/:f:/s/InsightsWEPA.digital-WEPA.digitalinternal/Er5bvpbhCZRMuah_7IR9LW4BxTCc__k_5pwJI1yDsoJ1_w?e=BdBCjm")

llm = get_llm()

db = get_local_database(set_serialization=True)

retriever = db.as_retriever()

template = return_prompt()

# Add history
question_summary_prompt = """Given a chat history and the latest user question \
    which  the context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otheriwse return it as is."""

question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", question_summary_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, question_prompt)




prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

qa_chain = create_stuff_documents_chain(llm, prompt2)


rag_chain_with_hisory = create_retrieval_chain(
    history_aware_retriever, qa_chain)

if "session_id" not in st.session_state:
    ts_str = str(pd.Timestamp.now())
    date_part, time_part = ts_str.split(" ")
    time_part = time_part.split(".")[0]
    new_id = date_part + time_part
    st.session_state["session_id"] = new_id + str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "conversation_df" not in st.session_state:
    st.session_state.conversation_df = pd.DataFrame(columns=["Day", "Response", "Question"])

def save_conversation():
    chat_history.update_one({"_id": st.session_state["session_id"]}, {"$set": {"chat_history": st.session_state.conversation_df.to_dict(orient="records")}})

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("User"):
            st.markdown(message.content)
    else:
        with st.chat_message("CEO"):
            st.markdown(message.content)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain_with_hisory,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)



with st.sidebar:
    st.title("Usage Metrics")

       
question = st.chat_input("Please Enter your questions...")
df = pd.DataFrame(columns=["Day","Response","Question"])
if question is not None and question != "":
    st.session_state.chat_history.append(HumanMessage(question))
    with st.chat_message("User"):
        st.markdown(question)

    with st.chat_message("CEO"):
        response = conversational_rag_chain.invoke(
            {"input": question},
            config={
                "configurable": {"session_id": "abc123"}
            }
        )["answer"]
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(response))
    
    new_row = {"Day": pd.Timestamp.now(), "Response": response, "Question": question}
    st.session_state.conversation_df = st.session_state.conversation_df._append(new_row, ignore_index=True)
