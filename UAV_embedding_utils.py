import os
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm
from langchain_community.vectorstores import Milvus

os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openai-proxy-6v0.pages.dev/v1"


embedding_model = OpenAIEmbeddings(model='text-embedding-3-large')


# 指定要遍历的目录
directory = "E:\desktop\无人机项目\\优先"


contents = []
# 遍历目录中的所有文件
for filename in os.listdir(directory):
    if filename.endswith(".pdf"):
        file_path = os.path.join(directory, filename)

        # 使用PyPDFLoader加载PDF文件
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        for page_num, page_content in enumerate(pages, start=1):
            contents.append(page_content)


for d in contents:
    d.page_content = d.page_content.replace('\n','').replace(' ', '')



db = Milvus(
        embedding_model,
        connection_args={"host": "ko.zhonghuapu.com", "port": "5530"},
        collection_name="UAV_db",
        auto_id=True
    )

db.add_documents(contents)

db
# Milvus.from_documents(
#         contents,
#         embedding_model,
#         collection_name="UAV_db",
#         connection_args={"host": "ko.zhonghuapu.com", "port": "5530"},
#         index_params={
#             'metric_type': 'COSINE',
#             'index_type': "IVF_FLAT",
#             'params': {"nlist": 256}
#         },
#     )
print("已经完成!")
