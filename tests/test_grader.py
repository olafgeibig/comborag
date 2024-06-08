import unittest
from unittest.mock import MagicMock
from comborag.grader import Grader
from types import SimpleNamespace
from langchain_community.chat_models.fake import FakeListChatModel

class TestGrader(unittest.TestCase):
    def setUp(self):
        # json = {"score": "yes"}
        json = """
```json
{"score": "yes"}
```
"""
        self.llm = FakeListChatModel(cache=False, responses=[[json]])
        self.mock_retriever = MagicMock()
        self.grader = Grader(self.llm, self.mock_retriever)

    # @patch('__main__.Document', autospec=True)
    def test_retrieval_grade(self):
        question = "What is the issue with my computer?"
        expected_response = {"score": "yes"}
        doc = SimpleNamespace(page_content="My computer has no drive and that's not okay.")
        docs = ["foo", doc]
        self.mock_retriever.invoke.return_value = docs
        # self.mock_llm.invoke.return_value = expected_response

        result = self.grader.retrieval_grade(question)
        print(result)
        self.assertEqual(result, expected_response)

    # def test_halluciantion_grade(self):
    #     generation = "The computer has no monitor."
    #     docs = "The servers are located in a datacenter in huge racks."
    #     expected_response = {"score": "no"}
    #     self.mock_llm.invoke.return_value = expected_response

    #     result = self.grader.halluciantion_grade(generation, docs)
    #     print(result)
    #     self.assertEqual(result, expected_response)

if __name__ == "__main__":
    unittest.main()
