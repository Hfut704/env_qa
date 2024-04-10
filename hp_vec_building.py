import json
import os

# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
os.environ["OPENAI_API_KEY"] = "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"
# os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.com/v1" # 配置中转代理，就不需要翻墙了，有风险！
os.environ["OPENAI_API_BASE"] = "https://openai-proxy-6v0.pages.dev/v1"
from echo_ai.embeddings import TextEmbedding
from langchain import FAISS
from langchain.schema.document import Document
from tqdm import tqdm
from langchain_community.vectorstores import Milvus
import json

import pandas as pd
from bs4 import BeautifulSoup


embedding_model = OpenAIEmbeddings(model='text-embedding-3-large')

def xls2doc(path, header_line, source, core_columns):
    full_info_docs = []
    key_info_docs = []
    df = pd.read_excel(path, header=header_line)
    list_of_dicts = df.to_dict(orient='records')
    for d in tqdm(list_of_dicts):
        for k in d:
            if type(d[k]) == str:
                soup = BeautifulSoup(d[k], 'html.parser')
                # 提取文本内容
                formatted_text = soup.get_text()
                formatted_text = formatted_text.replace(' ', "")
                d[k] = formatted_text

    for item in tqdm(list_of_dicts):
        full_info = ""
        key_part = ""
        for k in item:
            full_info += f"#{k}#\n"
            full_info += f"{item[k]}\n"

            if k in core_columns:
                key_part += f"#{k}#\n"
                key_part += f"{item[k]}\n"
        full_info_docs.append(Document(page_content=full_info, metadata={'source':source, 'data': json.dumps(item, ensure_ascii=False)}))
        key_info_docs.append(Document(page_content=key_part, metadata={'source':source,  'data': json.dumps(item, ensure_ascii=False)}))
    return full_info_docs, key_info_docs


def init_db(core_info_list, full_info_list):
    # 创建库
    Milvus.from_documents(
        core_info_list,
        embedding_model,
        collection_name="hb_key_info",
        connection_args={"host": "ko.zhonghuapu.com", "port": "5530"},
        index_params={
            'metric_type': 'COSINE',
            'index_type': "IVF_FLAT",
            'params': {"nlist": 256}
        },
    )
    Milvus.from_documents(
        full_info_list,
        embedding_model,
        collection_name="hb_full_info",
        connection_args={"host": "ko.zhonghuapu.com", "port": "5530"},
        index_params={
            'metric_type': 'COSINE',
            'index_type': "IVF_FLAT",
            'params': {"nlist": 256}
        },
    )


def add_new_to_db(core_info_list, full_info_list):
    full_info_db = Milvus(
        embedding_model,
        connection_args={"host": "ko.zhonghuapu.com", "port": "5530"},
        collection_name="hb_full_info",
        auto_id=True
    )

    full_info_db.add_texts([d.page_content for d in full_info_list], [d.metadata for d in full_info_list])
    key_info_db = Milvus(
        embedding_model,
        connection_args={"host": "ko.zhonghuapu.com", "port": "5530"},
        collection_name="hb_key_info",
        auto_id=True
    )
    key_info_db.add_texts([d.page_content for d in core_info_list], [d.metadata for d in core_info_list])


# # 读取 Excel 文件
# df = pd.read_excel('E:\desktop\环保项目\合规条款总库_数据.xlsx', header=1)
# df_1 = pd.read_excel('E:\desktop\环保项目\知识管理_数据.xlsx', header=1)
# df_2 = pd.read_excel('E:\desktop\环保项目\法规管理_数据.xlsx', header=1)

full_info_list, core_info_list = xls2doc('E:\desktop\环保项目\知识管理_数据.xlsx', header_line=1, source='知识管理', core_columns=['法规名称', '知识名称', '合规类别标签', '合规对象标签', '规定要求'])
init_db(core_info_list , full_info_list)

full_info_list, core_info_list = xls2doc('E:\desktop\环保项目\合规条款总库_数据.xlsx', header_line=1, source='合规条款总库', core_columns=['法规名称', '知识名称', '合规类别标签', '合规对象标签', '规定要求'])
add_new_to_db(core_info_list , full_info_list)

full_info_list, core_info_list = xls2doc('E:\desktop\环保项目\法规管理_数据.xlsx', header_line=1, source='法规管理', core_columns=['法规名称', '知识名称', '合规类别标签', '合规对象标签', '规定要求'])
add_new_to_db(core_info_list , full_info_list)



print('完成')

# db1 = FAISS.from_documents(key_info_docs, TextEmbedding())
# db1.save_local('vector_storage/hp_dbs/key_info_db')
# db2 = FAISS.from_documents(full_info_docs, TextEmbedding())
# db2.save_local('vector_storage/hp_dbs/full_info_db')
#
# db1
