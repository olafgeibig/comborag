from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from loguru import logger


class Indexer:
    def __init__(self, collection_name="rag-chroma"):
        self.vectorstore = self.create_vectorstore(collection_name)
        self.retriever = self.vectorstore.as_retriever()

    def create_vectorstore(self, collection_name):
        logger.info(f"Creating vectorstore with collection name: {collection_name}")
        return Chroma(
            collection_name=collection_name,
            embedding=GPT4AllEmbeddings(model_name="ggml-model-name"),
        )

    def index_urls(self, urls):
        logger.info(f"Indexing URLs: {urls}")
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Add to vectorDB
        self.vectorstore.add_documents(doc_splits)
