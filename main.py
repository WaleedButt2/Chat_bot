from typing import Any, Optional
from uuid import UUID
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from google.cloud import storage
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain

from langchain_core.outputs import LLMResult
load_dotenv()
from langchain.document_loaders import PDFMinerLoader
bucket_name = os.getenv("bucket_name")
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders.csv_loader import CSVLoader
import tempfile
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.current_line = initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token)
        self.current_line += token
        self.container.write(self.current_line)
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self.container.write('')
        return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    add_start_index = True,
)
db=None
st.title("Open AI Chat CSV - 🦜🦙")
response_container = st.container()
app = FastAPI()
st_callback = StreamHandler(st.empty())
current_chain = None
llm=None
# Initialize or load embeddings and history from session state
if "model_initialized" not in st.session_state:
    embeddings = OpenAIEmbeddings()
    st.session_state["model_initialized"] = True
    st.session_state["embeddings"] = embeddings
    st.session_state["llm"] = ChatOpenAI(model='gpt-3.5-turbo', streaming=True)
    llm= ChatOpenAI(model='gpt-3.5-turbo', streaming=True)
else:
    embeddings = st.session_state["embeddings"]
    llm=st.session_state["llm"]

if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Please upload a CSV or a PDF or I will crash. 🤗"]
if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]

with response_container:
    for i, (user_msg, ai_msg) in enumerate(zip(st.session_state['past'], st.session_state['generated'])):
        st.text_area("User", user_msg, key=f"user_msg_{i}", disabled=True)
        st.text_area("AI", ai_msg, key=f"ai_msg_{i}", disabled=True)

# File uploaders
uploaded_file_csv = st.sidebar.file_uploader("Upload CSV", type="csv")
if uploaded_file_csv:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file_csv.getvalue())
            tmp_file_path = tmp_file.name 
    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    #print(data)
    db = FAISS.from_documents(data,embeddings)
    retriever = db.as_retriever()
    current_chain = ConversationalRetrievalChain.from_llm(llm, retriever) 
uploaded_file_pdf = st.sidebar.file_uploader("Upload PDF - 🦜🦙", type="pdf")

if uploaded_file_pdf:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file_pdf.getvalue())
            tmp_file_path = tmp_file.name 
    loader = PDFMinerLoader(file_path=tmp_file_path)
    data = loader.load_and_split()
    #print(data)
    db = FAISS.from_documents(data,embeddings)
    retriever = db.as_retriever()
    current_chain = ConversationalRetrievalChain.from_llm(llm, retriever) 
if db==None:
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "{chat_history} You're a very knowledgeable Ai Bot of our company (Ab-Ark) and have great knowledge about the relevant stacks the compnay works on. (.Net,Ai,python,Django,Mern/Mean,Angular,Power BI)",
        ),
         ("human", "{query}"),
    ]
)
    output_parser = StrOutputParser()
    current_chain= LLMChain(llm=llm,prompt=prompt,output_parser=output_parser)
    # Query handling
def query_chain(query: str):
    if db!=None:
        result = current_chain({"question": query, "chat_history": st.session_state['history']}, callbacks=[st_callback])
        st.session_state['history'].append((query, result["answer"]))
        return result['answer']
    else:
        answer= current_chain.invoke({"query": query,"chat_history":st.session_state['history']})
        st.session_state['history'].append((query, answer))
        return answer

# User input and response
with st.form(key='my_form', clear_on_submit=True):
    user_input = st.text_input("Query:", placeholder="Talk to File data 👉 (:", key='input')
    submit_button = st.form_submit_button(label='Send')
    with response_container:
            st.text_area("User", user_input, key=f"user_msg_new", disabled=True)
if submit_button and user_input:
    output = query_chain(user_input)
    # Add to session state for display
    with response_container:
        st.text_area("AI", output, key=f"ai_msg_new", disabled=True)
    st.session_state['past'].append(user_input)
    #st.session_state['generated'].append(output)
    # Display the new messages
