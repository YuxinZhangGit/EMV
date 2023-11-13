"""
@文件    :data_process.py
@时间    :2022/04/21 13:27:59
@作者    :周恒
@版本    :1.0
@说明    :
"""


from typing import Any, Dict, List, Tuple, Union
from dataclasses import asdict, dataclass
import torch
from transformers.file_utils import PaddingStrategy
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer


@dataclass
class Inputfeature:
    sentence_a: str
    sentence_b: str
    sentence_c: str
    event_seq_a: List[str] = None
    event_seq_b: List[str] = None
    event_seq_c: List[str] = None
    entity_seq_a: List[int] = None
    entity_seq_b: List[int] = None
    entity_seq_c: List[int] = None


def process_event_list(event_list: List[Dict[str, Union[str, int]]]) -> List[str]:
    event_start = []
    for event in event_list:
        if len(event["event_type"]) > 0:
            event_start.append(
                (event["event_type"], event["trigger_start_index"]))
    if len(event_start) > 0:
        event_start.sort(key=lambda x: x[1])
        size = len(event_start)
        res = [event_start[0][0]]
        for i in range(1, size):
            if event_start[i][1] != event_start[i-1][1]:
                res.append(event_start[i][0])
        return res
    return []


def data_preprocess(raw_data: Dict[str, Union[str, List[str]]]) -> Inputfeature:
    a = raw_data["A"]
    b = raw_data["B"]
    c = raw_data["C"]
    res = Inputfeature(
        a["text"],
        b["text"],
        c["text"],
        process_event_list(a["event_list"]),
        process_event_list(b["event_list"]),
        process_event_list(b["event_list"])
    )
    return res


class Collator:
    def __init__(self,
                 tokenizer: BertTokenizer,
                 use_event_represent: bool,
                 use_text_represent: bool,
                 use_entity_represent: bool,
                 event_type2index: Dict[str, int] = None
                 ) -> None:
        self.tokenizer: BertTokenizer = tokenizer
        self.use_event_represent: bool = use_event_represent
        self.use_text_represent: bool = use_text_represent
        self.use_entity_represent: bool = use_entity_represent

        if self.use_event_represent:
            self.event_type2index: Dict[str, int] = event_type2index

    def __call__(self, batch_data: List[Inputfeature]) -> Tuple[torch.Tensor, ...]:
        a_event_sequence = torch.zeros([120],dtype=torch.uint8)
        a_text_input_ids = torch.zeros([120],dtype=torch.uint8)
        a_text_mask =torch.zeros([120],dtype=torch.uint8)
        a_entity_sequence=torch.zeros([120],dtype=torch.uint8)
        b_event_sequence =torch.zeros([120],dtype=torch.uint8)
        b_text_input_ids = torch.zeros([120],dtype=torch.uint8)
        b_text_mask =torch.zeros([120],dtype=torch.uint8)
        b_entity_sequence=torch.zeros([120],dtype=torch.uint8)
        c_event_sequence = torch.zeros([120],dtype=torch.uint8)
        c_text_input_ids = torch.zeros([120],dtype=torch.uint8)
        c_text_mask = torch.zeros([120],dtype=torch.uint8)
        c_entity_sequence=torch.zeros([120],dtype=torch.uint8)
        if self.use_text_represent:
            a_texts = []
            b_texts = []
            c_texts = []
            for inputfeature in batch_data:
                a_texts.append(inputfeature.sentence_a)
                b_texts.append(inputfeature.sentence_b)
                c_texts.append(inputfeature.sentence_c)
            a_tokenizer_output = self.tokenizer(
                a_texts,
                padding=PaddingStrategy.LONGEST,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
            b_tokenizer_output = self.tokenizer(
                b_texts,
                padding=PaddingStrategy.LONGEST,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
            c_tokenizer_output = self.tokenizer(
                c_texts,
                padding=PaddingStrategy.LONGEST,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
            a_text_input_ids, a_text_mask = a_tokenizer_output[
                "input_ids"], a_tokenizer_output["attention_mask"]
            b_text_input_ids, b_text_mask = b_tokenizer_output[
                "input_ids"], b_tokenizer_output["attention_mask"]
            c_text_input_ids, c_text_mask = c_tokenizer_output[
                "input_ids"], c_tokenizer_output["attention_mask"]
        if self.use_event_represent:
            a_eventid_seqs: List[List[int]] = []
            b_eventid_seqs: List[List[int]] = []
            c_eventid_seqs: List[List[int]] = []
            a_max_size = 1
            b_max_size = 1
            c_max_size = 1
            for inputfeature in batch_data:
                a_eventid_seqs.append([self.event_type2index[event]
                                      for event in inputfeature.event_seq_a])
                a_max_size = max(a_max_size, len(inputfeature.event_seq_a))
                b_eventid_seqs.append([self.event_type2index[event]
                                      for event in inputfeature.event_seq_b])
                b_max_size = max(b_max_size, len(inputfeature.event_seq_b))
                c_eventid_seqs.append([self.event_type2index[event]
                                      for event in inputfeature.event_seq_c])
                c_max_size = max(c_max_size, len(inputfeature.event_seq_c))
            for event_seqs, max_size in zip([a_eventid_seqs, b_eventid_seqs, c_eventid_seqs], [a_max_size, b_max_size, c_max_size]):
                for event_seq in event_seqs:
                    if len(event_seq) < max_size:
                        event_seq.extend(
                            [0 for i in range(max_size-len(event_seq))])
            a_event_sequence = torch.tensor(a_eventid_seqs, dtype=torch.long)
            b_event_sequence = torch.tensor(b_eventid_seqs, dtype=torch.long)
            c_event_sequence = torch.tensor(c_eventid_seqs, dtype=torch.long)

        if self.use_entity_represent:
            a_entityid_seqs:List[List[int]]=[]
            b_entityid_seqs:List[List[int]]=[]
            c_entityid_seqs:List[List[int]]=[]
            a_max_size = 1
            b_max_size = 1
            c_max_size = 1
            for inputfeature in batch_data:
                a_entityid_seqs.append([0]+list(map(lambda x:x+1,inputfeature.entity_seq_a)))
                a_max_size=max(a_max_size,len(a_entityid_seqs[-1]))
                b_entityid_seqs.append([0]+list(map(lambda x:x+1,inputfeature.entity_seq_b)))
                b_max_size=max(b_max_size,len(b_entityid_seqs[-1]))
                c_entityid_seqs.append([0]+list(map(lambda x:x+1,inputfeature.entity_seq_c)))
                c_max_size=max(c_max_size,len(c_entityid_seqs[-1]))
            for entityid_seqs,max_size in zip([a_entityid_seqs,b_entityid_seqs,c_entityid_seqs],[a_max_size,b_max_size,c_max_size]):
                for entityid_seq in entityid_seqs:
                    entityid_seq.extend([0 for i in range(max_size-len(entityid_seq))])
            a_entity_sequence=torch.tensor(a_entityid_seqs,dtype=torch.long)
            b_entity_sequence=torch.tensor(b_entityid_seqs,dtype=torch.long)
            c_entity_sequence=torch.tensor(c_entityid_seqs,dtype=torch.long)

        labels = torch.zeros([len(batch_data)], dtype=torch.long)

        return a_event_sequence,\
            a_text_input_ids,\
            a_text_mask,\
            a_entity_sequence,\
            b_event_sequence,\
            b_text_input_ids,\
            b_text_mask,\
            b_entity_sequence,\
            c_event_sequence,\
            c_text_input_ids,\
            c_text_mask,\
            c_entity_sequence,\
            labels
