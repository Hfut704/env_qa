import os
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm
from langchain_community.vectorstores import Milvus
import torch
from echo_ai.embeddings import *
from BARTScore.bart_score import BARTScorer


def get_sim(text_1, text_2):

    embed_1 = embedding_model.embed_query(text_1)
    embed_2 = embedding_model.embed_query(text_2)
    A = torch.tensor(embed_1)
    B = torch.tensor(embed_2)
    # sim = torch.nn.functional.cosine_similarity(A.unsqueeze(0), B.unsqueeze(0))

    print('#########################')
    # 1. 计算余弦相似度
    cosine_similarity = torch.nn.functional.cosine_similarity(A.unsqueeze(0), B.unsqueeze(0))
    print(f'余弦相似度: {cosine_similarity.item()}')

    # 2. 计算欧几里得距离
    euclidean_distance = torch.norm(A - B)
    print(f'欧几里得距离: {euclidean_distance.item()}')

    # 3. 计算曼哈顿距离
    manhattan_distance = torch.sum(torch.abs(A - B))
    print(f'曼哈顿距离: {manhattan_distance.item()}')

    # # 计算交集
    # intersection = torch.sum(torch.min(A, B))
    # # 计算重叠系数
    # overlap_coefficient = intersection / torch.min(torch.sum(A), torch.sum(B))
    # print(f'重叠系数: {overlap_coefficient.item()}')
    #
    # # 计算交集
    # intersection = torch.sum(torch.min(A, B))
    #
    # # 计算并集
    # union = torch.sum(torch.max(A, B))
    #
    # # 计算杰卡德相似度
    # jaccard_similarity = intersection / union
    # print(f'杰卡德相似度: {jaccard_similarity.item()}')

os.environ["OPENAI_API_KEY"] = "sk-proj-xMSZZcUlCWYZLr5BrlGmT3BlbkFJ8rhtn99XGA2R2LsvEtMP"
os.environ["OPENAI_API_BASE"] = "https://opeani-proxy-qpstwipccp.ap-southeast-1.fcapp.run/v1"


# embedding_model = OpenAIEmbeddings(model='text-embedding-3-large')

embedding_model = HuggingFaceTextEmbedding()



text_1 = """
吴信东，汉族，1963年9月出生于安徽省安庆市枞阳县项铺镇，数据挖掘研究与应用专家，俄罗斯工程院外籍院士，合肥工业大学教授、博士生导师。 [2] [6] [8-10]
吴信东于1984年获得合肥工业大学计算机科学与技术学士学位；1987年获得合肥工业大学计算机科学与技术硕士学位；1987年7月—1991年3月任合肥工业大学助教；1993年获得英国爱丁堡大学人工智能博士学位；1993年7月—2001年8月先后任职于澳大利亚和美国的3所大学；2001年9月任美国佛蒙特大学计算机科学系正教授；2001年9月—2010年6月任佛蒙特大学计算机科学系主任；2009年获得“国家自然科学基金海外及港澳学者合作研究基金（海外杰青）”资助 [9]；2022年当选为俄罗斯工程院外籍院士。 [5]
"""

text_2 = """
吴信东，1963年出生于安徽省安庆市枞阳县项铺镇，汉族。现任美国佛蒙特大学计算机科学系正教授及系主任。他曾在合肥工业大学担任助教，并在澳大利亚和美国的多所大学任职。吴信东获得了英国爱丁堡大学的人工智能博士学位，并获得国家自然科学基金海外及港澳学者合作研究基金（海外杰青）的资助。他还被授予俄罗斯工程院外籍院士的荣誉，同时担任博士生的导师。
"""

text_3 ="""
吴信东，汉族，出生于安徽省安庆市枞阳县项铺镇，目前担任美国佛蒙特大学计算机科学系的正教授及系主任。他获得了国家自然科学基金的海外及港澳学者合作研究基金（海外杰青），并被授予俄罗斯工程院外籍院士的荣誉。此外，吴信东还指导博士生，培养未来的科研人才。"""

text_4 = """
吴信东，汉族，1963年出生于安徽省安庆市枞阳县项铺镇。他获得了合肥工业大学的计算机科学与技术学士和硕士学位，并在该校任教，担任助教。
"""


text_5 = """
吴信东，汉族，1963年9月出生于安徽省安庆市枞阳县项铺镇。他于1984年获得合肥工业大学计算机科学与技术学士学位，随后在1987年获得该校的计算机科学与技术硕士学位。此后，他于1987年7月至1991年3月期间担任合肥工业大学的助教。
"""

text_6 = """
吴信东，汉族，1963年9月出生于安徽省安庆市枞阳县项铺镇。他于1984年获得合肥工业大学计算机科学与技术学士学位，1987年获得该校硕士学位。随后，他于1993年在英国爱丁堡大学获得人工智能博士学位。吴信东曾在澳大利亚和美国的三所大学任职，2001年起担任美国佛蒙特大学计算机科学系正教授，并于2001年至2010年期间担任系主任。2009年，他获得国家自然科学基金海外及港澳学者合作研究基金资助。2022年，吴信东当选为俄罗斯工程院外籍院士。
"""

text_7 = """
吴信东，汉族，生于1963年9月，来自安徽省安庆市枞阳县项铺镇。他于1987年获得合肥工业大学计算机科学与技术的学士和硕士学位，并在1987年7月至1991年3月期间担任合肥工业大学的助教。
"""

text_8 = """
吴信东，汉族，1963年9月出生于安徽省安庆市枞阳县项铺镇。他于1984年获得合肥工业大学计算机科学与技术学士学位，1987年获得硕士学位，并在同年开始担任合肥工业大学助教。1993年，他在英国爱丁堡大学获得人工智能博士学位，随后在澳大利亚和美国的三所大学任职。2001年9月，他成为美国佛蒙特大学计算机科学系的正教授，并于同年起担任系主任，直至2010年6月。2009年，他获得国家自然科学基金海外及港澳学者合作研究基金（海外杰青）的资助。2022年，他被授予俄罗斯工程院外籍院士的荣誉。
"""
get_sim(text_1, text_2)
get_sim(text_1, text_3)
get_sim(text_1, text_4)
get_sim(text_1, text_5)
get_sim(text_1, text_6)
get_sim(text_1, text_7)
get_sim(text_1, text_8)
bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
res = bart_scorer.score([text_2, text_3, text_4, text_5, text_6, text_7, text_8],[text_1] * 7, batch_size=8)
print(torch.exp(torch.tensor(res)))
res = bart_scorer.score([text_1] * 7, [text_2, text_3, text_4, text_5, text_6, text_7, text_8], batch_size=8)
print(torch.exp(torch.tensor(res)))
print(res)