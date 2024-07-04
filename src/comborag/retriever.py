from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from loguru import logger


class Retriever:
    def __init__(self, collection_name="rag-chroma"):
        self.vectorstore = self.create_vectorstore(collection_name)
        self.retriever = self.vectorstore.as_retriever()

    def create_vectorstore(self, collection_name):
        logger.info(f"Creating vectorstore with collection name: {collection_name}")
        return Chroma(
            collection_name=collection_name,
            embedding_function=GPT4AllEmbeddings(
                model_name="all-MiniLM-L6-v2.gguf2.f16.gguf"
            ),
        )

    def index_urls(self, urls):
        logger.info(f"Indexing URLs: {urls}")
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        # documents = "\n\n".join(doc.page_content for doc in docs_list)
        # print(docs_list)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Add to vectorDB
        self.vectorstore.add_documents(doc_splits)

    def retrieve(self, query):
        logger.info(f"Retrieving documents for query: {query}")
        return self.retriever.invoke(query)
