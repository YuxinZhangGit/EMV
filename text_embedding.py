from typing import List
import torch
from transformers import AutoModel, BertTokenizer
from transformers.file_utils import PaddingStrategy

from transformers.models.bert import BertForMaskedLM, BertModel, BertConfig
#conda环境下安装typing torch transformers
class TextEmbedder(object):
    def __init__(self,mlm_name_or_path:str,deviceId:str='cpu') -> None:
       self.tokenizer=BertTokenizer.from_pretrained(mlm_name_or_path)
       self.mlm: BertModel=AutoModel.from_pretrained(mlm_name_or_path)
       self.mlm.eval()
       self.device:torch.device=torch.device(deviceId)
       self.mlm=self.mlm.to(self.device)
    def getTextEmbedding(self,texts:List[str])->torch.Tensor:
        #分词
        
        tokenizerOutput:torch.Tensor=self.tokenizer(texts,padding=PaddingStrategy.LONGEST,return_tensors='pt')
        #batch_size * longestSentenceSize
        
        mlmOutput=self.mlm(**tokenizerOutput)

        #batch_size * 768
        cls_embedding:torch.Tensor=mlmOutput[0][:,0,:]

        return cls_embedding


if __name__=='__main__':
    embedder=TextEmbedder("hfl/chinese-roberta-wwm-ext") #写名称，BERT就是hfl/chinese-bert-wwm-ext 可以直接从网络上下载！
    texts=[
        "美国白宫新闻秘书珍·普萨基15日曾警告称，印度如果购买俄罗斯石油，就是站在了历史错误的一边。然而英媒发现，印度3月对俄罗斯的石油进口量正在激增。",
        "据英国《金融时报》18日报道，3月以来，俄罗斯每天向印度出口36万桶石油，几乎是2021年平均水平的四倍。大宗商品数据分析公司Kpler的数据显示，根据当前的发货时间表，俄罗斯3月的日均产量有望达到20.3万桶。"
    ] #demo
    embeddings=embedder.getTextEmbedding(texts)
    print(embeddings.shape)