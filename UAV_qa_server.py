import os

import uvicorn
from fastapi import FastAPI, Response,Query
from fastapi.responses import StreamingResponse
import markdown
from utils import *
from UAV_chatbot import *
import os


os.environ["OPENAI_API_BASE"] = my_args['openai_api_base']
os.environ["OPENAI_API_KEY"] = my_args['openai_api_key']
app = FastAPI()

# 初始化智能问答机器人
# chatbot = ZlChatBot().init_chatbot_from_faiss(
#     ['vector_storage/hp_dbs/full_info_db', 'vector_storage/hp_dbs/key_info_db'])

chatbot = UAV_Chatbot().init_chatbot_from_milvus(my_args['milvus_host'], port=my_args['milvus_port'],
                                                collections=['UAV_db'])


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
        return StreamingResponse(generate_json_stream_result(chatbot.query2kb_stream(req_data)),
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
    return AnswerResult(response=res)


@app.get("/v0/query2kb_stream")
async def query2kb_stream_v0(q: str, llm: str = Query("gpt")):
    """
        返回流式数据接口
        :param req_data:
        :return:
        """
    query = QueryRequest(question=q, llm=llm)
    res = "" #chatbot.get_from_cache(query.question)
    if res:
        # 如果存在缓存则以流式数据的方式返回数据。
        return StreamingResponse(generate_steam(res), media_type="text/html")
    else:
        # chatbot.get_stream返回的时字符流， generate_json_stream_result将字符流转换为json数据块流
        return StreamingResponse(chatbot.query2kb_stream(query),
                                 media_type="text/html")


@app.get("/v0/query2kb")
async def query2kb_v0(q: str,llm: str = Query("gpt")):
    """
    直接返回答案
    :param req_data:
    :return:
    """
    res = None  # chatbot.get_from_cache(q)
    if not res:
        query = QueryRequest(question=q, llm=llm)
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
    uvicorn.run( app=app, host="127.0.0.1", port=5550)