import unittest
from unittest.mock import MagicMock
from comborag.grader import Grader

class TestGrader(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_retriever = MagicMock()
        self.grader = Grader(self.mock_llm, self.mock_retriever)

    def test_retrieval_grade(self):
        question = "What is the issue with my computer?"
        expected_response = {"score": "yes"}
        self.mock_llm.invoke.return_value = expected_response

        result = self.grader.retrieval_grade(question)
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
