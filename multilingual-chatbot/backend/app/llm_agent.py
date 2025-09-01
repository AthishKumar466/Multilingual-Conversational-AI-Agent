import os
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

def get_llm(api_key: str = None):
    api_key = api_key or os.environ.get(OPENAI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError("OpenAI API key is not set.")
    return OpenAI(openai_api_key=api_key, temperature=0.2)

SYSTEM_PROMPT = """You are a helpful multilingual assistant."""

PROMPT_TEMPLATE = """
{system}
User: {user}
Assistant:
"""

def generate_reply(user_text: str, system: str = SYSTEM_PROMPT, api_key: str = None) -> str:
    llm = get_llm(api_key=api_key)
    prompt = PromptTemplate(input_variables=["system", "user"], template=PROMPT_TEMPLATE)
    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.run({"system": system, "user": user_text})
    return resp.strip()