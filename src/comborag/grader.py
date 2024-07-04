from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger


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
        self.answer_prompt = PromptTemplate(
            template="""You are a grader assessing whether an 
            answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
            useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            Here is the answer:
            \n ------- \n
            {generation} 
            \n ------- \n
            Here is the question: {question}""",
            input_variables=["generation", "question"],
        )
        self.llm = llm
        self.output_parser = JsonOutputParser()

    def retrieval_grade(self, question, documents):
        """
        A function to grade the retrieval based on the given question.

        Parameters:
            question: The question to be used for retrieval.

        Returns:
            The result of the retrieval grade.
        """
        # docs = self.retriever.invoke(question)
        # doc_txt = docs[1].page_content
        retrieval_grader = self.retrieval_prompt | self.llm | self.output_parser
        result = retrieval_grader.invoke({"question": question, "document": documents})
        logger.info(f"Retrieval grade: {question}: {result}")
        return result

    def halluciantion_grade(self, generation, documents):
        """
        A function to grade the hallucination based on the generation and documents input.

        Parameters:
            generation: The generated hallucination.
            documents: The documents used for grading.

        Returns:
            The result of invoking the hallucination grader with the generation and documents.
        """
        hallucination_grader = self.hallucination_prompt | self.llm | self.output_parser
        result = hallucination_grader.invoke(
            {"generation": generation, "documents": documents}
        )
        logger.info(f"Hallucination grade: {result}")
        return result

    def answer_grade(self, question, generation):
        """
        A function to grade the answer based on the question and generation input.

        Parameters:
            question: The question provided for grading.
            generation: The generated answer to the question.

        Returns:
            The result of invoking the answer grader with the question and generation.
        """
        answer_grader = self.answer_prompt | self.llm | self.output_parser
        result = answer_grader.invoke({"question": question, "generation": generation})
        logger.info(f"Answer grade: {question}: {result}")
        return result
