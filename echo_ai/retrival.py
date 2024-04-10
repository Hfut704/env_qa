from typing import List
from langchain import FAISS
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Milvus

class MyRetrival:
    """
    自定义的检索器，从单个或多个知识库中检索相关的知识
    """

    def __init__(self):
        self.embeddings = []
        self.vector_dbs = []
        self.embed_db_dirs = []
        pass

    def init_from_faiss_dbs(self, db_dirs: List[str], embeddings: List[Embeddings]):
        """
        从多个本地的faiss向量库中初始化知识库
        :param db_dirs:
        :param embeddings:
        :return:
        """
        self.embed_db_dirs = db_dirs
        self.embeddings += embeddings
        for db_dir, embedding_model in zip(db_dirs, embeddings):
            self.vector_dbs.append(FAISS.load_local(db_dir, embeddings=embedding_model))



    def init_from_milvus(self, host, port, collections: List[str], embeddings: List[Embeddings]):
        """
        使用milvus初始化检索器
        @param host:
        @param port:
        @param collections:
        @param embeddings:
        """
        for embedding, collection in zip(embeddings, collections):
            db = Milvus(
                embedding,
                connection_args={"host": host, "port": port},
                collection_name=collection,
            )
            self.vector_dbs.append(db)


    def get_relevant_documents(self, query: str, k=4):
        """
        从构建的多个向量库中检索出相关的知识，并进行过滤，去除冗余信息
        """
        docs_list = []
        for db in self.vector_dbs:

            ds = db.similarity_search_with_score(query, k)
            if len(ds) == 0:
                raise ValueError(
                    "vector store is empty."
                )
            else:
                docs_list.append(ds)
        return docs_list


if __name__ == '__main__':

    pass
