from flask import Flask, request, render_template
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import timeit
import sys
load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY','14ad6742-a148-486b-a115-0a5d50765c4d')
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV', 'gcp-starter')

#***Extract Data From the PDF File***
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents

extracted_data=load_pdf_file(data='data/')

# #print(data)


#***Split the Data into Text Chunks****
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=text_split(extracted_data)
print("Length of Text Chunks", len(text_chunks))

#***Download the Embeddings from Hugging Face***
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

# start = timeit.default_timer()
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="langchainpinecones"

#Creating Embeddings for Each of The Text Chunks
#docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

#If we already have an index we can load it like this
docsearch=Pinecone.from_existing_index(index_name, embeddings)



prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="models\llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':70,
                          'temperature':0.9})

print("here")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/op", methods=["POST"])
def qa():
    qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={'k': 2}),return_source_documents=True, chain_type_kwargs=chain_type_kwargs)
    # print(request.form.get("user_input"))
    user_input = str(request.form.get("user_input")) 
    print(user_input)
    result = qa({"query": user_input})
    print("Response : ", result["result"])
    return render_template("qa.html", result=result)
