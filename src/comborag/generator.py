from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

class Generator:
    def __init__(self, llm, retriever):
        self.prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
            Question: {question} 
            Context: {context} 
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["question", "document"],
        )
        self.llm = llm
        self.retriever = retriever
                
    def generate(self, question):
        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = self.prompt | self.llm | StrOutputParser()

        docs = self.retriever.invoke(question)
        generation = rag_chain.invoke({"context": docs, "question": question})
        return generation