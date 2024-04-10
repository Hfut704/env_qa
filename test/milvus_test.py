import json
import os

from langchain_community.embeddings import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "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"
# os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.com/v1" # 配置中转代理，就不需要翻墙了，有风险！
os.environ["OPENAI_API_BASE"] = "https://openai-proxy-6v0.pages.dev/v1"
from echo_ai.embeddings import TextEmbedding
from langchain import FAISS
from langchain.schema.document import Document
from  tqdm import  tqdm
from langchain_community.vectorstores import Milvus
import json

import pandas as pd
from bs4 import BeautifulSoup


embeddings = OpenAIEmbeddings()

vector_db = Milvus(
    embeddings,
    connection_args={"host": "ko.zhonghuapu.com", "port": "5530"},
    collection_name="hb_full_info",
)
res = vector_db.similarity_search_with_score("排污量",k=5)
res
