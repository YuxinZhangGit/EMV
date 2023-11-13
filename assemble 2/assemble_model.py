"""
@文件    :assemble_model.py
@时间    :2022/04/20 18:47:53
@作者    :周恒
@版本    :1.0
@说明    :
"""

from typing import Iterator, Union, Tuple, Dict
from sklearn.metrics import accuracy_score
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Embedding, LSTM, Dropout, CosineSimilarity, LayerNorm, TransformerEncoder,TransformerEncoderLayer,PairwiseDistance
from transformers.models.bert import BertModel
from transformers.models.roberta import RobertaModel
from transformers.models.longformer import LongformerModel

from trainer import Trainer


class EventRepresentModel(torch.nn.Module):
    def __init__(self, event_type_size: int, event_vec_dim: int, rnn_dim: int = 256, rnn_num_layers=1) -> None:
        super().__init__()
        """第一个为占位符"""
        self.event_type_size: int = event_type_size
        self.event_vec_dim: int = event_vec_dim
        self.rnn_num_layers: int = rnn_num_layers
        self.event_embedding: Embedding = Embedding(
            event_type_size+1, event_vec_dim, padding_idx=0)

        self.rnn = LSTM(event_vec_dim, rnn_dim,
                        self.rnn_num_layers, batch_first=True)
        self.drop_out = Dropout()
        self.output_dim: int = rnn_dim*rnn_num_layers

    def forward(self, event_sequence: torch.Tensor) -> torch.Tensor:
        self.rnn.flatten_parameters()
        batch_size, sequence = event_sequence.shape
        """batch_size * sequence * event_vec_dim"""
        embeddings = self.event_embedding(event_sequence)
        # event_mask_repeated=torch.where(event_sequence.bool(),1,0).reshape([batch_size,sequence,1]).repeat([1,1,self.event_vec_dim]).float()

        lstm_ouptut = self.rnn(embeddings)
        """num_layers * batch_size *  rnn_dim"""
        h_n: torch.Tensor = lstm_ouptut[1][0]
        """batch_size * num_layers *  rnn_dim"""
        h_n = h_n.transpose(0, 1)
        """batch_size * event_vec_dim"""
        res = h_n.reshape([batch_size, -1])
        return res


class TextRepresentModel(torch.nn.Module):
    def __init__(self, mlm_model: Union[BertModel, RobertaModel, LongformerModel]) -> None:
        super().__init__()
        self.mlm: Union[BertModel, RobertaModel, LongformerModel] = mlm_model
        self.output_dim: int = self.mlm.config.hidden_size

    def forward(self, text_input_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        sequence_output = self.mlm(
            input_ids=text_input_ids, attention_mask=text_mask)[0]
        cls: torch.Tensor = sequence_output[:, 0, :]
        return cls


class EntityRepresentModel(torch.nn.Module):
    def __init__(self, entity_embedding: Embedding,layer_num:int=6,nhead:int=8) -> None:
        super().__init__()
        self.entity_type_size = entity_embedding.num_embeddings
        self.entity_vec_dim = entity_embedding.embedding_dim

        self.entity_embedding: Embedding = entity_embedding
        self.cls_embedding = Parameter(torch.randn([self.entity_vec_dim],dtype=torch.float32))
        """第一个是占位符"""
        self.encoder:TransformerEncoder=TransformerEncoder(
            TransformerEncoderLayer(self.entity_vec_dim,nhead=nhead,batch_first=True),num_layers=layer_num
        )
        self.output_dim: int = self.entity_vec_dim

    # def train(self: 'EntityRepresentModel', mode: bool = True) -> 'EntityRepresentModel':
    #     super().train(mode)
        # self.entity_embedding.train(False)
        # self.entity_embedding.requires_grad_(False)

        # return self
        # emb=emb.detach_()
    # def parameters(self, recurse: bool = True) -> Iterator[torch.nn.parameter.Parameter]:
    #     return self.transformer.parameters(recurse)

    def forward(self, entity_sequence: torch.Tensor) -> torch.Tensor:
        emb: torch.Tensor = self.entity_embedding(entity_sequence)
        emb[:, 0, :] = self.cls_embedding
        res: torch.Tensor = self.encoder(emb)

        return res[:, 0, :]


class AssembleModel(torch.nn.Module):
    def __init__(self, use_event_represent: bool, use_text_represent: bool, use_entity_represent: bool, ** kwargs) -> None:
        super().__init__()
        self.use_event_represent: bool = use_event_represent
        self.use_text_represent: bool = use_text_represent
        self.use_entity_represent: bool = use_entity_represent
        self.feature_size = 0
        if self.use_event_represent:
            self.event_represent_model: EventRepresentModel = kwargs["event_represent_model"]
            self.feature_size += self.event_represent_model.output_dim
        if self.use_text_represent:
            self.text_represent_model: TextRepresentModel = kwargs["text_represent_model"]
            self.feature_size += self.text_represent_model.output_dim
        if self.use_entity_represent:
            self.entity_represent_model: EntityRepresentModel = kwargs["entity_represent_model"]
            self.feature_size += self.entity_represent_model.output_dim

        self.cosSim = CosineSimilarity(1)
        self.distance=PairwiseDistance()
        self.layer_norm = LayerNorm([self.feature_size], eps=1e-5)

    def forward(
        self,
        a_event_sequence: torch.Tensor, a_text_input_ids: torch.Tensor, a_text_mask: torch.Tensor, a_entity_sequence: torch.Tensor,
        b_event_sequence: torch.Tensor, b_text_input_ids: torch.Tensor, b_text_mask: torch.Tensor, b_entity_sequence: torch.Tensor,
        c_event_sequence: torch.Tensor, c_text_input_ids: torch.Tensor, c_text_mask: torch.Tensor, c_entity_sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # batch_size = 0
        a_feature, b_feature, c_feature = None, None, None
        if self.use_event_represent:
            batch_size, _ = a_event_sequence.shape
            a_feature = self.event_represent_model(a_event_sequence)
            b_feature = self.event_represent_model(b_event_sequence)
            c_feature = self.event_represent_model(c_event_sequence)
        if self.use_text_represent:
            if a_feature is None:
                batch_size, _ = a_text_input_ids.shape
                a_feature = self.text_represent_model(
                    a_text_input_ids, a_text_mask)
                b_feature = self.text_represent_model(
                    b_text_input_ids, b_text_mask)
                c_feature = self.text_represent_model(
                    c_text_input_ids, c_text_mask)
            else:
                a_cls = self.text_represent_model(
                    a_text_input_ids, a_text_mask)
                b_cls = self.text_represent_model(
                    b_text_input_ids, b_text_mask)
                c_cls = self.text_represent_model(
                    c_text_input_ids, c_text_mask)
                a_feature = torch.cat([a_feature, a_cls], dim=1)
                b_feature = torch.cat([b_feature, b_cls], dim=1)
                c_feature = torch.cat([c_feature, c_cls], dim=1)
        if self.use_entity_represent:
            batch_size, _ =a_entity_sequence.shape
            a_entity_cls = self.entity_represent_model(a_entity_sequence)
            b_entity_cls = self.entity_represent_model(b_entity_sequence)
            c_entity_cls = self.entity_represent_model(c_entity_sequence)
            if a_feature is None:
                a_feature = a_entity_cls
                b_feature = b_entity_cls
                c_feature = c_entity_cls
            else:
                a_feature = torch.cat([a_feature, a_entity_cls], dim=1)
                b_feature = torch.cat([b_feature, b_entity_cls], dim=1)
                c_feature = torch.cat([c_feature, c_entity_cls], dim=1)
        a_feature = self.layer_norm(a_feature)
        b_feature = self.layer_norm(b_feature)
        c_feature = self.layer_norm(c_feature)
        simAB: torch.Tensor = -(a_feature-b_feature).norm(dim=1).reshape([-1])
        simAC: torch.Tensor = -(a_feature-c_feature).norm(dim=1).reshape([-1])
        return a_feature, b_feature, c_feature, simAB, simAC


class BatchCalLossFunc:
    def __init__(self, margin: float) -> None:
        self.margin = margin
        self.tripleML = torch.nn.TripletMarginLoss(margin=self.margin)
        # self.mseLoss=torch.nn.MSELoss(reduction="none")
        # self.relu=torch.nn.ReLU()
        # self.cosEmbeddingLoss=torch.nn.CosineEmbeddingLoss(margin=0.5,reduction="none")
        # self.marginRankingLoss=MarginRankingLoss(alpha)

    def __call__(self, labels: torch.Tensor, preds: Tuple[torch.Tensor, ...], trainer: Trainer) -> torch.Tensor:
        a_feature, b_feature, c_feature, simAB, simAC = preds

        batch_size = labels.shape[0]
        # disAB=self.mseLoss(clsA,clsB).sum(1)
        # disAC=self.mseLoss(clsA,clsC).sum(1)
        # loss=self.relu(simPrefixC*disAB+simPrefixB*disAC)
        loss = self.tripleML(a_feature, b_feature, c_feature)
        return loss.sum()/batch_size


def batch_metrics_func(labels: torch.Tensor, preds: Tuple[torch.Tensor, ...],  metrics: Dict[str, Union[bool, int, float]], trainer: Trainer):
    a_feature, b_feature, c_feature, simAB, simAC = preds

    label = labels
    batchSize = label.shape[0]

    pred = torch.where(simAB > simAC, 0, 1).cpu()
    label = label.cpu()
    # print()
    print(f"\npreds:{pred.numpy()}")
    print(f"labels:{label.numpy()}")
    acc = accuracy_score(label.cpu().numpy(), pred.cpu().numpy())
    label = label.cpu()
    batch_metrics = {"acc": acc}
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
    preds = metrics["preds"]
    labels = metrics["labels"]
    acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    res = {"acc": acc}
    return res


def batch_forward_func(batch_data: Tuple[torch.Tensor, ...], trainer):

    a_event_sequence,\
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
        labels = batch_data

    a_event_sequence,\
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
        labels =\
        a_event_sequence.cuda(trainer.device, non_blocking=True),\
        a_text_input_ids.cuda(trainer.device, non_blocking=True),\
        a_text_mask.cuda(trainer.device, non_blocking=True),\
        a_entity_sequence.cuda(trainer.device, non_blocking=True),\
        b_event_sequence.cuda(trainer.device, non_blocking=True),\
        b_text_input_ids.cuda(trainer.device, non_blocking=True),\
        b_text_mask.cuda(trainer.device, non_blocking=True),\
        b_entity_sequence.cuda(trainer.device, non_blocking=True),\
        c_event_sequence.cuda(trainer.device, non_blocking=True),\
        c_text_input_ids.cuda(trainer.device, non_blocking=True),\
        c_text_mask.cuda(trainer.device, non_blocking=True),\
        c_entity_sequence.cuda(trainer.device, non_blocking=True),\
        labels.cuda(trainer.device, non_blocking=True)
    model: AssembleModel = trainer.model
    a_feature, b_feature, c_feature, simAB, simAC = model(
        a_event_sequence,
        a_text_input_ids,
        a_text_mask,
        a_entity_sequence,
        b_event_sequence,
        b_text_input_ids,
        b_text_mask,
        b_entity_sequence,
        c_event_sequence,
        c_text_input_ids,
        c_text_mask,
        c_entity_sequence
    )
    return labels, (a_feature, b_feature, c_feature, simAB, simAC)


def get_optimizer(model: AssembleModel, lr: float):
    if model.use_entity_represent:
        params=[]
        if model.use_event_represent:
            params.append({"params":model.event_represent_model.parameters()})
        if model.use_text_represent:
            params.append({"params":model.text_represent_model.parameters()})
        params.append({"params":model.entity_represent_model.cls_embedding})
        params.append({"params":model.entity_represent_model.encoder.parameters()})
        optimizer = torch.optim.AdamW(params, lr=lr)
    else:
        optimizer = torch.optim.AdamW({"params":model.parameters()}, lr=lr)
    return optimizer
