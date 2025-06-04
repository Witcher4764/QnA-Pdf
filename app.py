import streamlit as st
import os
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts.chat import ChatPromptTemplate , MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

## Getting embeddings from local model running via ollama
embeds = OllamaEmbeddings(model="mxbai-embed-large:335m")

## set up Streamlit 
st.title("Q&A On a Pdf")
st.write("Upload a Pdf and ask questions from the content")

## Input the Groq API Key Or you can use local llm using Ollama
api_key=st.text_input("Enter your Groq API key:",type="password")

if api_key:

    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

    session_id=st.text_input("Session ID",value="0")
    
    if 'store' not in st.session_state:
        st.session_state.store={}

    user_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
    
    if user_files:
        documents=[]
        for user_file in user_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(user_file.getvalue())
                file_name=user_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

    
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        doc_splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=doc_splits, embedding=embeds)
        retriever = vectorstore.as_retriever()    

        system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    
        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        qa_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,qa_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                }, 
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")

