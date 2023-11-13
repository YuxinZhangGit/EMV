from ctypes import sizeof
from turtle import shape
from typing import List
import torch
#from transformers import AutoModel, BertTokenizer
#from transformers.file_utils import PaddingStrategy
#from transformers.models.bert import BertModel
import text_embedding as txt_emb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_score(txt_embs, labels):
    cnt = 0
    correct = 0
    #print(txt_embs)
    #print(txt_embs[0][1].detach().numpy())
    #计算余弦相似度
    for i in range(len(txt_embs)):
        cnt += 1
        #print(txt_embs[i][0])
        simA_B = int(cosine_similarity(txt_embs[i][0].detach().numpy().reshape(1,-1),txt_embs[i][1].detach().numpy().reshape(1,-1)))
        simA_C = int(cosine_similarity(txt_embs[i][0].detach().numpy().reshape(1,-1),txt_embs[i][2].detach().numpy().reshape(1,-1)))
        if simA_B > simA_C:
            if labels[i] == 1:
                correct += 1
        else:
            if labels[i] == 2:
                correct += 1
    return 1.0*correct/cnt

if __name__=='__main__':
    texts=[
        [
        "美国白宫新闻秘书珍·普萨基15日曾警告称，印度如果购买俄罗斯石油，就是站在了历史错误的一边。然而英媒发现，印度3月对俄罗斯的石油进口量正在激增。",
        "据英国《金融时报》18日报道，3月以来，俄罗斯每天向印度出口36万桶石油，几乎是2021年平均水平的四倍。大宗商品数据分析公司Kpler的数据显示，根据当前的发货时间表，俄罗斯3月的日均产量有望达到20.3万桶。",
        "美国白宫新闻秘书珍·普萨基15日曾警告称，印度如果购买俄罗斯石油，就是站在了历史错误的一边。然而英媒发现，印度3月对俄罗斯的石油进口量正在激增。"
    ],
        [
        "美国白宫新闻秘书珍·普萨基15日曾警告称，印度如果购买俄罗斯石油，就是站在了历史错误的一边。然而英媒发现，印度3月对俄罗斯的石油进口量正在激增。",
        "据英国《金融时报》18日报道，3月以来，俄罗斯每天向印度出口36万桶石油，几乎是2021年平均水平的四倍。大宗商品数据分析公司Kpler的数据显示，根据当前的发货时间表，俄罗斯3月的日均产量有望达到20.3万桶。",
        "美国白宫新闻秘警告印度，如果印度购买俄罗斯石油，就是与美国为敌。然而英媒发现，印度正在大量进口俄罗斯石油。"
    ]
    ]
    labels=[2,2]
    embedder=txt_emb.TextEmbedder("hfl/chinese-bert-wwm-ext")
    txt_embs = []
    for item in texts:
        txt_embs.append(embedder.getTextEmbedding(item))
    #print(txt_embs) 

    res = get_score(txt_embs,labels)
    print(res)