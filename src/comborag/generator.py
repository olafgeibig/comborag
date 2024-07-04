from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger


class Generator:
    def __init__(self, llm):
        self.prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise
            Question: {question} 
            Context: {context}""",
            input_variables=["question", "context"],
        )
        self.llm = llm

    def generate(self, question, documents):
        logger.info(f"Generating answer for question: {question}")
        rag_chain = self.prompt | self.llm | StrOutputParser()
        generation = rag_chain.invoke({"context": documents, "question": question})
        return generation
