from comborag.grader import Grader
from comborag.retriever import Retriever
from comborag.generator import Generator
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

def main():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Create a retriever and index URLs
    retriever = Retriever()
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]    
    retriever.index_urls(urls)
    question = "agent memory"
    
    
    generator = Generator(llm=llm, retriever=retriever.retriever)
    answer = generator.generate(question)

    grader = Grader(llm=llm, retriever=retriever.retriever)
    print(grader.halluciantion_grade(question))

if __name__ == "__main__":
    main()
