
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

source_docs = [
    Document(
        page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]}
    ) for doc in knowledge_base
]

docs_processed = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(source_docs)[:1000]

embedding_model = HuggingFaceEmbeddings("thenlper/gte-small")
vectordb = FAISS.from_documents(
    documents=docs_processed,
    embedding=embedding_model
)

knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

all_sources = list(set([doc.metadata["source"] for doc in docs_processed]))
print(all_sources)

['blog', 'optimum', 'datasets-server', 'datasets', 'transformers', 'course',
'gradio', 'diffusers', 'evaluate', 'deep-rl-class', 'peft',
'hf-endpoints-documentation', 'pytorch-image-models', 'hub-docs']