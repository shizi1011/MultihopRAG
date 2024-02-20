import os
import streamlit as st
import tempfile
from localllm import Loadllm
from langchain_community.document_loaders import PyMuPDFLoader
from streamlit_chat import message
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

DB_FAISS_PATH = 'vectorstore/db_faiss'
DB_CHROMA_PATH = 'vectorstore/chroma_db'


class FileIngestor:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file

    def handlefileandingest(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(self.uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(file_path=tmp_file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # Create embeddings using Nomic Embeddings
        # embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

        # Create embeddings using OpenAI Embeddings
        # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Create embeddings using Huggingface Bge Embeddings
        embeddings_model_name = "BAAI/bge-small-en-v1.5"
        embeddings_model_kwargs = {"device": "cpu"}
        embeddings_encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=embeddings_model_name, model_kwargs=embeddings_model_kwargs, encode_kwargs=embeddings_encode_kwargs
        )

        # Create a FAISS vector store and save embeddings
        # db = FAISS.from_documents(docs, embeddings)
        # db.save_local(DB_FAISS_PATH)

        # Create a Chroma vector store and save embeddings
        # db = Chroma.from_documents(
        #     docs, embeddings, persist_directory=DB_CHROMA_PATH)

        from chromadb.errors import InvalidDimensionException

        try:
            db = Chroma.from_documents(
                docs, embeddings, persist_directory=DB_CHROMA_PATH)
        except InvalidDimensionException:
            Chroma().delete_collection()
            db = Chroma.from_documents(
                docs, embeddings, persist_directory=DB_CHROMA_PATH)


        # Create retrivever from vector database
        retriever = db.as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Load the language model
        llm = Loadllm.load_llm()

        # Create a conversational chain
        # chain = ConversationalRetrievalChain.from_llm(
        #     llm=llm, retriever=db.as_retriever())

        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \ If you don't know the answer, just say that you don't know. \ Use three sentences maximum and keep the answer concise.\

        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        def contextualized_question(input: dict):
            if input.get("chat_history"):
                return contextualize_q_chain
            else:
                return input["question"]

        rag_chain = (
            RunnablePassthrough.assign(
                context=contextualized_question | retriever | format_docs
            )
            | qa_prompt
            | llm
        )
        # Function for conversational chat
        # def conversational_chat(query):
        #     result = chain(
        #         {"question": query, "chat_history": st.session_state['history']})
        #     st.session_state['history'].append((query, result["answer"]))
        #     return result["answer"]

        def conversational_chat(query):
            result = rag_chain.invoke(
                {"question": query, "chat_history": st.session_state['history']})

            st.session_state['history'].extend(
                [HumanMessage(content=query), result])
            return result.content

        # Initialize chat history
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        # Initialize messages
        if 'generated' not in st.session_state:
            st.session_state['generated'] = [
                "Hello ! Ask me about " + self.uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]

        # Create containers for chat history and user input
        response_container = st.container()
        container = st.container()

        # User input form
        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input(
                    "Query:", placeholder="Talk to PDF data 🧮", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        # Display chat history
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(
                        i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(
                        i), avatar_style="thumbs")
