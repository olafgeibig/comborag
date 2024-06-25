from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings


class Indexer:
    def __init__(self):
        self.vectorstore = self.create_vectorstore()
        self.retriever = self.vectorstore.as_retriever()

    def create_vectorstore(self):
        return Chroma(
            collection_name="rag-chroma",
            embedding=GPT4AllEmbeddings(),
        )

    def index_urls(self, urls):
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Add to vectorDB
        self.vectorstore.add_documents(doc_splits)
