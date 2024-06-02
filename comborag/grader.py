from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class Grader:
    def __init__(self, llm):
        self.retrieval_prompt = PromptTemplate(
            template="""system
            You are a grader assessing relevance 
            of a retrieved document to a user question. If the document contains keywords related to the user question, 
            grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explaination. \n
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question}
            """,
            input_variables=["question", "document"],
        )
        self.hallucination_prompt = PromptTemplate(
            template="""system You are a grader assessing whether 
            an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate 
            whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
            single key 'score' and no preamble or explanation.
            user
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}""",
            input_variables=["generation", "documents"],
        )
        self.llm = llm
        self.output_parser = JsonOutputParser()

    def retrieval_grade(self, question):
        # docs = retriever.invoke(question)
        # doc_txt = docs[1].page_content
        doc_txt = "My computer has no drive and that's not okay."
        retrieval_grader = self.retrieval_prompt | self.llm | self.output_parser
        return retrieval_grader.invoke({"question": question, "document": doc_txt})
    
    def halluciantion_grade(self, question, docs):
        hallucination_grader = self.hallucination_prompt | self.llm | self.output_parser
        return hallucination_grader.invoke({"question": question, "documents": docs})
