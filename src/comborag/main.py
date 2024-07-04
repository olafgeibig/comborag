from comborag.grader import Grader
from comborag.retriever import Retriever
from comborag.generator import Generator
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import tiktoken

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
    
    docs = retriever.retrieve(question)
    # print(docs)
    
    # Count tokens in retrieved documents using tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    for doc in docs:
        if isinstance(doc, str):  # Ensure doc is a string
            tokens = enc.encode(doc)
            total_tokens += len(tokens)
    print(f"Total tokens in retrieved documents: {total_tokens}")
    
    generator = Generator(llm=llm)
    answer = generator.generate(question, docs)
    print(answer)
    grader = Grader(llm=llm)
    print(grader.halluciantion_grade(question, docs))
    print(grader.answer_grade(question, answer))

if __name__ == "__main__":
    main()
