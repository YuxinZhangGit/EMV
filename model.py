'''
@文件    :model.py
@时间    :2022/03/31 19:10:51
@作者    :周恒
@版本    :1.0
@说明    :
'''



from ctypes import sizeof
from typing import Dict, Tuple,Union
import torch
from torch import Tensor
from torch.nn import CosineSimilarity,Module,MarginRankingLoss,CosineEmbeddingLoss,TripletMarginLoss
from transformers.models.bert import BertModel
from trainer import Trainer
from sklearn.metrics import recall_score, f1_score, precision_score,accuracy_score
class BertBaselineModel(Module):
    def __init__(self,mlm_name_or_path:str) -> None:
        super().__init__()
        self.bert:BertModel=BertModel.from_pretrained(mlm_name_or_path)
        self.cosSim=CosineSimilarity(1)
    def forward(
        self,tokensA:Tensor,tokensB:Tensor,tokensC:Tensor,
        maskA:Tensor,maskB:Tensor,maskC:Tensor
    )->Tuple[Tensor,Tensor]:
        batchSize,sequenceSize=maskA.shape
        # batchSize * 768
        outputA=self.bert(input_ids=tokensA,attention_mask=maskA)
        clsA:Tensor=outputA[0][:,0,:]
        clsB:Tensor=self.bert(input_ids=tokensB,attention_mask=maskB)[0][:,0,:]
        clsC:Tensor=self.bert(input_ids=tokensC,attention_mask=maskC)[0][:,0,:]
        
        simAB:Tensor=self.cosSim(clsA,clsB).reshape([batchSize])
        simAC:Tensor=self.cosSim(clsA,clsC).reshape([batchSize])
        return clsA,clsB,clsC,simAB,simAC
class BatchCalLossFunc:
    def __init__(self,alpha:float) -> None:
        self.alpha=alpha 
        self.tripleML=TripletMarginLoss()
        # self.mseLoss=torch.nn.MSELoss(reduction="none")
        # self.relu=torch.nn.ReLU()
        # self.cosEmbeddingLoss=torch.nn.CosineEmbeddingLoss(margin=0.5,reduction="none")
        # self.marginRankingLoss=MarginRankingLoss(alpha)
    def __call__(self,labels:Tuple[torch.Tensor, ...], preds: Tuple[torch.Tensor, ...], trainer:Trainer)->Tensor:
        clsA,clsB,clsC,simAB,simAC=preds
        # 正=-1 负=1
        label,simPrefixB,simPrefixC=labels
        
        batchSize=label.shape[0]
        # disAB=self.mseLoss(clsA,clsB).sum(1)
        # disAC=self.mseLoss(clsA,clsC).sum(1)
        # loss=self.relu(simPrefixC*disAB+simPrefixB*disAC)
        loss=self.tripleML(clsA,clsB,clsC)
        return loss.sum()/batchSize
        # return self.marginRankingLoss(simAB,simAC,torch.where(label==0,torch.ones([batchSize],device=label.device,dtype=torch.long),-torch.ones([batchSize],device=label.device,dtype=torch.long)).to(label.device))
        # return loss
def batch_metrics_func(labels:Tuple[torch.Tensor, ...], preds: Tuple[torch.Tensor, ...],  metrics: Dict[str, Union[bool, int, float]], trainer:Trainer):
    clsA,clsB,clsC,simAB,simAC=preds
    # 正=-1 负=1
    label,simPrefixB,simPrefixC=labels
    batchSize=label.shape[0]

    pred=torch.where(simAB>simAC,torch.zeros([1],device=label.device,dtype=torch.long),torch.ones([1],device=label.device,dtype=torch.long)).cpu()
    
    acc=accuracy_score(label.cpu().numpy(),pred.cpu().numpy())
    label=label.cpu()
    batch_metrics={"acc":acc}
    if "labels" in metrics:
        metrics["labels"] = torch.cat([metrics["labels"], label], dim=0)
    else:
        metrics["labels"] = label
    if "preds" in metrics:
        metrics["preds"] = torch.cat(
            [metrics["preds"], pred], dim=0)
    else:
        metrics["preds"] = pred
    
    return metrics, batch_metrics

def metrics_cal_func(metrics: Dict[str, torch.Tensor]):
    preds=metrics["preds"]
    labels=metrics["labels"]
    acc=accuracy_score(labels.cpu().numpy(),preds.cpu().numpy())
    res={"acc":acc}
    return res

def batch_forward_func(batch_data: Tuple[torch.Tensor, ...], trainer):
    tokensA,tokensB,tokensC,maskA,maskB,maskC,labels,simPrefixB,simPrefixC=batch_data
  
    tokensA,tokensB,tokensC,maskA,maskB,maskC,labels,simPrefixB,simPrefixC=\
        tokensA.cuda(trainer.device,non_blocking=True),\
        tokensB.cuda(trainer.device,non_blocking=True),\
        tokensC.cuda(trainer.device,non_blocking=True),\
        maskA.cuda(trainer.device,non_blocking=True),\
        maskB.cuda(trainer.device,non_blocking=True),\
        maskC.cuda(trainer.device,non_blocking=True),\
        labels.cuda(trainer.device,non_blocking=True),\
        simPrefixB.cuda(trainer.device,non_blocking=True),\
        simPrefixC.cuda(trainer.device,non_blocking=True)

    clsA,clsB,clsC,simAB,simAC=trainer.model(tokensA,tokensB,tokensC,maskA,maskB,maskC)
    return (labels,simPrefixB,simPrefixC),(clsA,clsB,clsC,simAB,simAC)

def get_optimizer(model:BertBaselineModel,lr:float):
    optimizer=torch.optim.AdamW([{"params":model.parameters()}],lr=lr)
    return optimizer