import os

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import markdown
from utils import *
from chatbot import *

os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openai-proxy-6v0.pages.dev/v1"

app = FastAPI()

# 初始化智能问答机器人
# chatbot = ZlChatBot().init_chatbot_from_faiss(
#     ['vector_storage/hp_dbs/full_info_db', 'vector_storage/hp_dbs/key_info_db'])

chatbot = HB_Chatbot().init_chatbot_from_milvus("ko.zhonghuapu.com", port='5530',
                                                collections=['hb_full_info', 'hb_key_info'])


def generate_steam(text: str):
    """
    产生一个字符流的数据
    :param text:
    :return:
    """
    for c in text:
        yield c
        time.sleep(0.0001)


def generate_json_stream_result(text_stream):
    """
    将text的字符流转为json数据块流
    :param text_stream:
    :return:
    """
    for chunk in text_stream:
        yield json.dumps(dict(StreamResult(block=chunk)), ensure_ascii=False) + '\n'
    yield json.dumps(dict(StreamResult(block='[END]', end=True)), ensure_ascii=False) + '\n'


@app.post("/v1/query2kb_stream")
async def query2kb_stream(req_data: QueryRequest):
    """
    返回流式数据接口
    :param req_data:
    :return:
    """

    res = chatbot.get_from_cache(req_data.question)
    if res:
        # 如果存在缓存则以流式数据的方式返回数据。
        return StreamingResponse(generate_json_stream_result(generate_steam(res)), media_type="application/json")
    else:
        # chatbot.get_stream返回的时字符流， generate_json_stream_result将字符流转换为json数据块流
        return StreamingResponse(generate_json_stream_result(chatbot.get_stream(req_data)),
                                 media_type="application/json")


@app.route("/v1/query2kb", methods=['GET', 'POST'])
async def query2kb(req_data: QueryRequest):
    """
    直接返回答案
    :param req_data:
    :return:
    """
    res = chatbot.get_from_cache(req_data.question)
    if not res:
        res = chatbot.query2kb(req_data)
        # query = req_data.question
        # # 检索文本相关的问答对
        # text_relevant_docs = chatbot.retrival.get_relevant_documents(query=query)
        #
        # # 获取得到两种方式相关的问答对后，合并
        # docs = text_relevant_docs
        # # 数据后处理
        # relevant_docs = chatbot.post_progress_data(docs)
        #
        # res = "\n\n".join(relevant_docs)#chatbot.query2kb(req_data)
        # print(res)
    return AnswerResult(response=res)


@app.get("/v0/query2kb_stream")
async def query2kb(q: str):
    """
        返回流式数据接口
        :param req_data:
        :return:
        """
    query = QueryRequest(question=q)
    res = chatbot.get_from_cache(query.question)
    if res:
        # 如果存在缓存则以流式数据的方式返回数据。
        return StreamingResponse(generate_steam(res), media_type="text/html")
    else:
        # chatbot.get_stream返回的时字符流， generate_json_stream_result将字符流转换为json数据块流
        return StreamingResponse(chatbot.get_stream(query),
                                 media_type="text/html")


@app.get("/v0/query2kb")
async def query2kb(q: str):
    """
    直接返回答案
    :param req_data:
    :return:
    """
    res = None  # chatbot.get_from_cache(q)
    if not res:
        query = QueryRequest(question=q)
        res = chatbot.query2kb(query)
        html_text = markdown.markdown(res)
        response = Response(content=html_text, media_type="text/html")
        return response


@app.get("/test")
async def test():
    """
    测试接口
    :return:
    """
    return "连接成功"


if __name__ == '__main__':
    uvicorn.run(app='qa_server:app', host="127.0.0.1", port=5500, reload=False)


