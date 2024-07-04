from comborag.grader import Grader
from comborag.retriever import Retriever
from comborag.generator import Generator
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import tiktoken
from loguru import logger
import sys

load_dotenv()  # take environment variables from .env.

logger.remove()
logger.add(sys.stderr, level="INFO") 

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
    logger.debug("That's it, beautiful and simple logging!")
    
    docs = retriever.retrieve(question)
    # print(docs)

    # Count tokens in retrieved documents using tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o")
    documents = "\n\n".join(doc.page_content for doc in docs)
    tokens = len(enc.encode(documents))
    print(f"Total tokens in retrieved documents: {tokens}")

    generator = Generator(llm=llm)
    answer = generator.generate(question, docs)
    print(answer)
    grader = Grader(llm=llm)
    print(grader.halluciantion_grade(question, docs))
    print(grader.answer_grade(question, answer))


if __name__ == "__main__":
    main()
