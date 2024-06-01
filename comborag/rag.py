from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

class Rag:
    def __init__(self, llm):
        self.prompt = PromptTemplate(
            template="""system You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise
            Question: {question} 
            Context: {context}""",
            input_variables=["question", "document"],
        )
        self.llm = llm

    def query(self, question, docs):
        """
        Executes a query using the RAG chain.

        Args:
            question (str): The question to be answered.
            docs (List[str]): The list of documents to be used as context.

        Returns:
            str: The answer to the question.

        Raises:
            None
        """
        rag_chain = self.prompt | self.llm | StrOutputParser()
        return rag_chain.invoke({"context": docs, "question": question})

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
