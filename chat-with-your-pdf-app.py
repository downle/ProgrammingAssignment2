import os
import openai
import sys
from dotenv import load_dotenv
from pathlib import Path  # Python 3.6+ only
import gradio as gr
# Huggingface models
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader
# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
# OpenAI llm
from langchain.chat_models import ChatOpenAI
# Build prompt
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import random
import time
import shutil


#env_path = Path('.') / 'environ_vars.env'
#load_dotenv(dotenv_path=env_path)
# Print variable FOO to test if this worked
#print(os.environ.get('FOO')) # Returns 'BAR'

openai.api_key  = os.environ['OPENAI_API_KEY']
hf_token  = os.environ['HUGGINGFACEHUB_API_TOKEN']


def augment_llm_with_docs(files):
    from langchain.document_loaders import PyPDFLoader
    
    # get file paths
    file_paths = [file.name for file in files]

    # Load PDF docs
    loaders = [PyPDFLoader(file) for file in file_paths]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # Split
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 400,
        chunk_overlap = 60
    )

    splits = text_splitter.split_documents(docs)

    from langchain.vectorstores import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings

    persist_directory_openai = 'docs/chroma/openai/'
    #!rm -rf ./docs/chroma/openai

    try:
        shutil.rmtree(persist_directory_openai)
        print("Directory removed successfully")
    except FileNotFoundError:
        print("/docs/chroma/openai directory does not exist")
        pass

    openai_embeddings = OpenAIEmbeddings()

    # Finally we make our Index (i.e. Vector Database) using chromadb and the open ai embeddings LLM
    openai_vectordb = Chroma.from_documents(
        documents=splits,
        embedding=openai_embeddings,
        persist_directory=persist_directory_openai
    ) # Chromadb index

    # OpenAI llm
    from langchain.chat_models import ChatOpenAI
    import datetime
    current_date = datetime.datetime.now().date()
    if current_date < datetime.date(2023, 9, 2):
        openai_llm_name = "gpt-3.5-turbo-0301"
    else:
        openai_llm_name = "gpt-3.5-turbo"

    openai_llm = ChatOpenAI(model_name=openai_llm_name, temperature=0)

    # Build prompt
    from langchain.prompts import PromptTemplate
    template = """Consider the following pieces of context. Decide whether they are relevant to answering the question at the end. If they are relevant, use them to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Think carefully about this. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    # Memory
    from langchain.memory import ConversationBufferMemory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Conversational Retrieval chain
    from langchain.chains import ConversationalRetrievalChain
    retriever=openai_vectordb.as_retriever()
    global qa
    qa = ConversationalRetrievalChain.from_llm(
        openai_llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        memory=memory
    )


def respond(message, chat_history):
    result = qa({"question": message})
    bot_message = result['answer']
    chat_history.append((message, bot_message))
    #time.sleep(2)
    return "", chat_history

with gr.Blocks() as demo:
    upload_pdfs = gr.File(file_types=[".pdf"], file_count="multiple")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton([msg, chatbot, upload_pdfs])

    upload_pdfs.upload(augment_llm_with_docs, inputs=upload_pdfs)
    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    
    gr.Markdown("""For example pdf documents, in the files section, you can go to the "docs/finance" folder 
    and choose the file(s) there to upload. \n With the week 1 intro file an example of a question you
    can ask is ***'What are some prerequisites of this class?'*** and a potential follow up question could be 
    ***'Okay, how about the main topics this class covers?'*** With all 3 files uploaded, you could
    first ask these questions, then dive deeper into the topics such as asking to know more about the time
    value of money.""")

demo.launch()
