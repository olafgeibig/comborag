from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


class Router:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""You are an expert at routing a 
            user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
            prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
            in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
            or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
            no premable or explaination. Question to route: {question}""",
            input_variables=["question"],
        )

    def router(self, question):
        router = self.prompt | self.llm | JsonOutputParser()
        return router.invoke({"question": question})
