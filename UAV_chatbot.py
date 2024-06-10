import json
import threading
from typing import List
from config import *
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from echo_ai.retrival import MyRetrival
import tiktoken
from utils import *
class UAV_Chatbot:
    """
    一个基于知识库的问答机器人的核心步骤包括：
    1）检索： 从知识库中检索相关的知识。
    2）后处理： 对检索得到的知识进行一些处理，比如，过滤，去重，召回，格式话等。
    3）推理： 从得到的知识中推理出答案。
    """

    def __init__(self):
        self.retrival = MyRetrival()  # 检索器，从单个获取多个知识库中检索相关的文本
        self.llm_name = my_args['openai_model'] if my_args['openai_model'] else 'gpt-3.5-turbo'
        self.llm = ChatOpenAI(
            model=self.llm_name,
            temperature=0,
        )  ## temperature越低回答越准确，越高创造性越强
        self.stream_llm = None
        self.cache = {}
        # self.xf_llm = SparkLLM(temperature=0.1, version=3.1)
        self.token_encoder = tiktoken.encoding_for_model(my_args['openai_model'])
        self.max_token = int(my_args['max_token']) if my_args['max_token'] and my_args['max_token'].isdigit() else 4096

    def init_chatbot_from_milvus(self , host, port, collections: List[str]):
        """
        从milvus中初始化chatbot
        @param host:
        @param port:
        @param collections:
        @return:
        """
        embedd_models = []
        for collection in collections:
            embedd_models.append(OpenAIEmbeddings(model=my_args['embedding_model']))
        self.retrival.init_from_milvus(host, port, collections, embedd_models)
        return self

    def init_chatbot_from_faiss(self, db_dirs: List[str]):
        """
        初始化中梁项目的知识库
        :param db_dirs: 向量库所在的目录
        :return:
        """
        embeddings = [OpenAIEmbeddings(), OpenAIEmbeddings()]
        self.retrival.init_from_faiss_dbs(db_dirs, embeddings)
        return self

    def get_from_cache(self, query: str):
        """
        从缓存中取回答
        :param query:
        :return:
        """
        hash_code = hash(query)
        return self.cache.get(hash_code)

    def post_progress_data(self, docs_list: List[List]):
        """
        主要对从多个向量库得到的信息进行一个过滤去冗余 和 对json格式的数据处理成字符串格式的。
        :param docs_list:
        :return:
        """
        docs = [d for ds in docs_list for d in ds]
        docs = sorted(docs, key=lambda x: x[1],reverse=True)
        res = set()
        for doc in docs:
            item = json.loads(doc[0].metadata['data'])
            full_info = ""
            for k in item:
                full_info += f"##{k}##\n"
                full_info += f"{item[k]}\n"
            res.add(full_info)
        return list(res)


    def query2kb(self, query_data: QueryRequest):
        """
        从本地知识库中检索相关知识并回答问题
        :param query_data:
        :param llm:
        :return:
        """
        query = query_data.question
        prompt, content = self.prepare_prompt(query)
        ans = self.llm.invoke(prompt).content
        self.cache[hash(query)] = ans
        content.replace('\n', '\n >')
        ans += f"<br><br><br><br><br><br><br><br><br><br><br><br><br><br> <br><br><br><br><br><br><br>  <h>参考信息</h> <br> \n{content}\n"
        return ans

    def query2kb_stream(self, query_data: QueryRequest):
        """
        从本地知识库中检索相关知识并回答问题,流式回答
        :param query_data:
        :param llm:
        :return:
        """
        query = query_data.question
        prompt, content = self.prepare_prompt(query)
        ans = self.llm.invoke(prompt).content
        for chunk in ans:
            yield chunk

    def prepare_prompt(self, query: str):
        # 检索文本相关的问答对
        text_relevant_docs = self.retrival.get_relevant_documents(query=query)

        # 数据后处理
        relevant_docs =  [d[0].page_content for ds in text_relevant_docs for d in ds ]
        content = ""
        content_token_len = 0
        for d in relevant_docs:
            content_token_len += len(self.token_encoder.encode(d))
            if content_token_len > self.max_token:
                break
            content += d
        prompt = f"""

KNOWLEDGE:
<knowledge>\n{content}\n</knowledge>


请根据"KNOWLEDGE"中的信息回答问题。
您需要仔细考虑您的答案，并确保它基于上下文，如果能够给出依据，则在回答中给出你得出答案的依据。
如果根据"KNOWLEDGE"无法得到答案，则回答“根据知识库的内容暂时无法得到准确答案。”
请不要回答从"KNOWLEDGE"无法推断出的内容。
必须使用{"Chinese"}进行回应。 
QUERY:
{query}
"""
        return prompt, content


