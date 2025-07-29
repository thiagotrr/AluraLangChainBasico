# Case: agente de viagem com cadeia de LangChain 
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA #Deprecated
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.globals import set_debug

set_debug(True)

llm = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.7
)

carregadores = [
    PyPDFLoader("GTB_standard_Nov23.pdf"),
    PyPDFLoader("GTB_gold_Nov23.pdf"),
    PyPDFLoader("GTB_platinum_Nov23.pdf"),
    PyPDFLoader("GTB_black_Nov23.pdf"),
]

documentos = []
for carregador in carregadores:
    documentos.extend(carregador.load())

quebrador = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #1000 caracteres.
textos = quebrador.split_documents(documentos)
#print(textos)

embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
db = FAISS.from_documents(textos, embeddings)

# Cadeia de perguntas
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

pergunta = "Como devo proceder caso um item que comprei seja roubado?"

resultado = qa_chain.invoke({"query": pergunta})
print(resultado)