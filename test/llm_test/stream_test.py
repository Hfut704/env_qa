# from langchain.chat_models.openai import ChatOpenAI
from langchain_community.chat_models.openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(temperature=0.0, model='gpt-4o-mini')
res = llm.invoke('你好')

