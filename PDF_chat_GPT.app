import gradio as gr
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

def loading_pdf(): return 'Loading...'

def pdf_changes(pdf_doc, repo_id):
    loader = OnlinePDFLoader(pdf_doc.name)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2096, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceHubEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={'temperature': 0.5, 'max_new_tokens': 2096})
    global qa 
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True)
    return "Ready"

def add_text(history, text):
    history = history + [(text, None)]
    return history, ''

def bot(history):
    response = infer(history[-1][0])
    history[-1][1] = response['result']
    return history

def infer(question):
    query = question
    result = qa({'query': query})
    return result

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
    <h1>Chat with PDF</h1>
"""

with gr.Blocks(css=css, theme='Taithrah/Minimal') as demo:
    with gr.Column(elem_id='col-container'):
        gr.HTML(title)

    with gr.Column():
        pdf_doc = gr.File(label='Upload a PDF', file_types=['.pdf'])
        repo_id = gr.Dropdown(label='LLM', 
                              choices=[
                                  'mistralai/Mistral-7B-Instruct-v0.1', 
                                  'HuggingFaceH4/zephyr-7b-beta', 
                                  'meta-llama/Llama-2-7b-chat-hf', 
                                  '01-ai/Yi-6B-200K'
                                  'cognitivecomputations/dolphin-2.5-mixtral-8x7b'
                              ],
                             value='mistralai/Mistral-7B-Instruct-v0.1')
        with gr.Row():
            langchain_status = gr.Textbox(label='Status', placeholder='', interactive=False)
            load_pdf = gr.Button('Load PDF to LangChain')

        chatbot = gr.Chatbot([], elem_id='chatbot')#.style(height=350)
        question = gr.Textbox(label='Question', placeholder='Type your query')
        submit_btn = gr.Button('Send')

    repo_id.change(pdf_changes, inputs=[pdf_doc, repo_id], outputs=[langchain_status], queue=False)
    load_pdf.click(pdf_changes, inputs=[pdf_doc, repo_id], outputs=[langchain_status], queue=False)
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(bot, chatbot, chatbot)
    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(bot, chatbot, chatbot)

demo.launch()
