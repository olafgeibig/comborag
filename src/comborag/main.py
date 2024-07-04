from comborag.grader import Grader
from comborag.indexer import Indexer
from comborag.generator import Generator
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

def main():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Create an indexer and index URLs
    indexer = Indexer()
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]    
    indexer.index_urls(urls)
    question = "agent memory"
    
    
    generator = Generator(llm=llm, retriever=indexer.retriever)
    answer = generator.generate(question)

    grader = Grader(llm=llm, retriever=indexer.retriever)
    print(grader.halluciantion_grade(question))

if __name__ == "__main__":
    main()
