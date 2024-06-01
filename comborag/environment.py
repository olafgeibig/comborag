import getpass
import os

def set_environment_variables():
    def _set_if_undefined(var: str):
        if not os.environ.get(var):
            os.environ[var] = getpass.getpass(f"Please provide your {var}")

    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("LANGCHAIN_API_KEY")
    _set_if_undefined("TAVILY_API_KEY")

    # Optional, add tracing in LangSmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"