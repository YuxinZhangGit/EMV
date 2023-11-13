
from ctypes import Union
from typing import List, Tuple
from dataclasses import asdict, dataclass
import torch
from transformers.file_utils import PaddingStrategy
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
@dataclass
class Inputfeature:
    sentenceA:str 
    sentenceB:str 
    sentenceC:str 
    label:str


class Collator:
    def __init__(self,tokenizer:BertTokenizer) -> None:
        self.tokenizer=tokenizer
    
    def __call__(self, data:List[Inputfeature]) ->Tuple[torch.Tensor,...]:
        batch_size=len(data)
        texts=[]
        labels=[]
        simPrefixB,simPrefixC=[],[]
        for inputfeature in data:
            texts.append(inputfeature.sentenceA)
            if inputfeature.label=='B':
                texts.append(inputfeature.sentenceB)
                texts.append(inputfeature.sentenceC)
                labels.append(0)
                simPrefixB.append(-1)
                simPrefixC.append(1)
            else:
                texts.append(inputfeature.sentenceC)
                texts.append(inputfeature.sentenceB)
                labels.append(0)
                simPrefixB.append(-1)
                simPrefixC.append(1)
        labels=torch.tensor(labels,dtype=torch.long)
        simPrefixB,simPrefixC=torch.tensor(simPrefixB,dtype=torch.float),torch.tensor(simPrefixC,dtype=torch.float)
        output=self.tokenizer(
            texts,
            padding=PaddingStrategy.LONGEST,
            truncation=True,
            max_length=512,
            return_attention_mask=True
            ) 
        input_ids=output["input_ids"]
        masks=output["attention_mask"]
        res=[]
        tokensA=torch.tensor([input_ids[i*3] for i in range(batch_size)],dtype=torch.long)
        tokensB=torch.tensor([input_ids[i*3+1] for i in range(batch_size)],dtype=torch.long)
        tokensC=torch.tensor([input_ids[i*3+2] for i in range(batch_size)],dtype=torch.long)
        maskA=torch.tensor([masks[i*3] for i in range(batch_size)],dtype=torch.long)
        maskB=torch.tensor([masks[i*3+1] for i in range(batch_size)],dtype=torch.long)
        maskC=torch.tensor([masks[i*3+2] for i in range(batch_size)],dtype=torch.long)

        return tokensA,tokensB,tokensC,maskA,maskB,maskC,labels,simPrefixB,simPrefixC