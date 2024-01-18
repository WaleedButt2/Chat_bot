from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from dotenv import load_dotenv, dotenv_values
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains import ConversationalRetrievalChain
import importlib
import os
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import LlamaCpp
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PDFMinerLoader
import streamlit as st
from langchain.callbacks import get_openai_callback
import openai
load_dotenv()
config = dotenv_values(".env")
class Chatbot:
    def __init__(self,path=None):
        openai.api_key = config["OPENAI_API_KEY"]
        os.environ['OPENAI_API_KEY']=openai.api_key
        #print(openai.api_key)
        self.embeddings = OpenAIEmbeddings()
        self.llm =LlamaCpp(
        model_path="./llama-2-7b-chat.Q5_K_S.gguf",
        n_gpu_layers=config["n_gpu_layers"],
        n_batch=config["n_batch"],
        n_ctx=4096,
        f16_kv=True,
        verbose=True  # Verbose is required to pass to the callback manager
    )
        #self.llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")
        self.memory = ConversationBufferWindowMemory(k=3, memory_key="conversation_history", input_key="question")
        self.chain = self._prepare_chain()
    def _create_db(self,file_path,type):
        docs=None
        if type=='pdf':
            loader = PDFMinerLoader(file_path)
            docs=loader.load_and_split()
        elif type=='csv':
            loader=CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
            docs=loader.load()
        if docs==None:
            return
        #st.write(docs)
        return FAISS.from_documents(docs,self.embeddings)
    def _loadVectorStores(self,name='test'):
            dbs=FAISS.load_local(f'{name}',self.embeddings)
            return dbs
    def _prepare_chain(self,name=None):
        if name!=None:
            db = self._loadVectorStores(self,name)
        else:
            db=None
        prompt = """Conversation_History :   {conversation_history}\nContext:{context}\n Answer the following query keeping the context in mind 
                                                  if query is out of context do not try to guess answers just say its not within the context.\nIf no Context is given
                                                  behave like a normal Chatbot.
                                                  """
        prompt = PromptTemplate(
            template=prompt,
            input_variables=["context", "conversation_history"],
            )
        system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
        human_template = """ Question: {question}\nBot: """
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt,
            human_message_prompt]
        )
        return  load_qa_chain(llm=self.llm, chain_type="stuff", verbose=False,
                                        prompt=chat_prompt, memory=self.memory)
        
    def get_query_response(self,query,k=3,db=None):
        #if path!=None:
            #db=self._loadVectorStores(path)
        #else:
            #db=None
        with get_openai_callback() as cb:
            context = [] 
            if db!=None:
                context_docs = db.similarity_search_with_score(query,k=k)
                docs = sorted(context_docs, key=lambda x: x[1])
                for doc, score in docs:
                    # print(score)
                    doc.metadata['score'] = score
                    context.append(doc)
            response = self.chain({"question": query, "input_documents": context})
            print(response)
            st.write(self.chain.memory.load_memory_variables({}))
            st.write(f"Total Tokens: {cb.total_tokens}")
            st.write(f"Prompt Tokens: {cb.prompt_tokens}")
            st.write(f"Completion Tokens: {cb.completion_tokens}")
            st.write(f"Total Cost (USD): ${cb.total_cost}")
            return (response["output_text"],context)