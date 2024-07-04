from comborag.grader import Grader
from comborag.indexer import Indexer
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

def main():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    grader = Grader(llm=llm)
    question = "agent memory"
    print(grader.retrieval_grade(question))

    # Create an indexer and index URLs
    indexer = Indexer()
    urls = ["http://example.com/page1", "http://example.com/page2"]
    indexer.index_urls(urls)

if __name__ == "__main__":
    main()
