from comborag.grader import Grader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

def main():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    grader = Grader(llm=llm)
    question = "agent memory"
    print(grader.retrieval_grade(question))


if __name__ == "__main__":
    main()
