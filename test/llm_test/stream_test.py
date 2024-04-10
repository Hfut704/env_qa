# from langchain.chat_models.openai import ChatOpenAI
from langchain_community.chat_models.openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openai-proxy-6v0.pages.dev/v1"

llm = ChatOpenAI(temperature=0.0, model='gpt-4-0125-preview', openai_api_base="https://openai-proxy-6v0.pages.dev/v1")
for chunk in llm.stream("你好"):
    chunk
    print(chunk, end="", flush=True)

