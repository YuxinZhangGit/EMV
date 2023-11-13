import argparse
import json
from assemble_model import *
from data_process import *
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer
import os
import torch
THIS_FOLDER = os.path.split(os.path.realpath(__file__))[0]


def init_embedding(emb_map: Dict[int, torch.Tensor])->Embedding:
    type_size, vec_size = len(emb_map), emb_map[0].shape[0]
    res = torch.nn.Embedding(type_size+1, vec_size, padding_idx=0)
    # res.weight.requires_grad = False
    weight = torch.zeros([type_size+1, vec_size],
                         dtype=torch.float32, requires_grad=False)
    for k, v in emb_map.items():
        weight[k+1] = v
    res.weight.data = weight
    assert res.weight.data[1].norm() == emb_map[0].norm()
    return res


parser = argparse.ArgumentParser()
parser.add_argument("--main_device", type=int, default=0)
parser.add_argument("--device_ids", type=str, default="0,1")
parser.add_argument("--batch_size", type=int, default=600)
parser.add_argument("--num_workers", type=int, default=12)
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--mlm_name_or_path", type=str,
                    default="../../roberta-chinese")
parser.add_argument("--gradient_accumulate", type=int, default=1)
parser.add_argument("--data_folder", type=str, default="data/scm")
parser.add_argument("--use_event_represent",
                    action="store_true", default=False)
parser.add_argument("--use_text_represent",
                    action="store_true", default=False)
parser.add_argument("--use_entity_represent",
                    action="store_true", default=True)
parser.add_argument("--event_vec_dim", type=int, default=768)
parser.add_argument("--rnn_dim", type=int, default=768)
parser.add_argument("--rnn_layer_num", type=int, default=1)
# parser.add_argument("--")
if __name__ == '__main__':
    args = parser.parse_args()

    """一些常规的设置"""
    dev = torch.device(args.main_device)
    device_ids = list(map(lambda x: int(x), args.device_ids.split(",")))
    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.learning_rate
    epochs = args.epochs
    mlm_name_or_path = args.mlm_name_or_path
    gradient_accumulate = args.gradient_accumulate
    data_folder: str = args.data_folder
    """任务特定设置"""
    use_event_represent = args.use_event_represent
    use_text_represent = args.use_text_represent
    use_entity_represent = args.use_entity_represent
    event_vec_dim = args.event_vec_dim
    rnn_dim = args.rnn_dim
    rnn_layer_num = args.rnn_layer_num

    for arg in args._get_kwargs():
        print(arg)

    train_dataset: List[Inputfeature] = []
    test_dataset: List[Inputfeature] = []
    event_type_to_id: Dict[str, int] = {}
    # event_id_to_type:Dict[int,str]={}
    with open(os.path.join(data_folder, "train.json"), "r") as f:
        js = json.load(f)
        for obj in js:
            train_dataset.append(data_preprocess(obj))
    with open(os.path.join(data_folder, "test.json"), "r") as f:
        js = json.load(f)
        for obj in js:
            test_dataset.append(data_preprocess(obj))

    with open(os.path.join(THIS_FOLDER, "event_type_to_id.json"), "r") as f:
        event_type_to_id = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("../../roberta-chinese")
    collator = Collator(tokenizer, use_event_represent,
                        use_text_represent, use_entity_represent, event_type_to_id)
    batch_cal_loss_func = BatchCalLossFunc(1.0)

    text_represent_model = None
    event_represent_model = None
    entity_represent_model = None
    if use_text_represent:
        mlm = AutoModel.from_pretrained(mlm_name_or_path)
        text_represent_model = TextRepresentModel(mlm)
    if use_event_represent:
        event_represent_model = EventRepresentModel(len(
            event_type_to_id), event_vec_dim, rnn_dim=rnn_dim, rnn_num_layers=rnn_layer_num)
    if use_entity_represent:
        """读取entity embedding"""
        entity_emb_map: Dict[int, torch.Tensor] = {}
        with open(os.path.join(data_folder, "entity_embedding.json"), "r") as f:
            for line in f:
                item = json.loads(line)
                entity_emb_map[item["index"]] = torch.tensor(
                    item["emb"], dtype=torch.float32)

        """读取entity index"""
        with open(os.path.join(data_folder, "entityindex_seq_train_list.json"), "r") as f:
            it = iter(train_dataset)
            for line in f:
                inputfeature = next(it)
                js = json.loads(line)
                inputfeature.entity_seq_a = js["A"]
                inputfeature.entity_seq_b = js["B"]
                inputfeature.entity_seq_c = js["C"]
        with open(os.path.join(data_folder, "entityindex_seq_test_list.json"), "r") as f:
            it = iter(test_dataset)
            for line in f:
                inputfeature = next(it)
                js = json.loads(line)
                inputfeature.entity_seq_a = js["A"]
                inputfeature.entity_seq_b = js["B"]
                inputfeature.entity_seq_c = js["C"]
        entity_emb=init_embedding(emb_map=entity_emb_map)
        entity_represent_model=EntityRepresentModel(entity_emb)
    model = AssembleModel(use_event_represent, use_text_represent, use_entity_represent,
                          event_represent_model=event_represent_model, text_represent_model=text_represent_model, entity_represent_model=entity_represent_model)
    optimizer = get_optimizer(model, learning_rate)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    metrics = []

    train_dataset_sampler = RandomSampler(train_dataset)
    test_dataset_sampler = SequentialSampler(test_dataset)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        output_dir="saved",
        training_dataset=train_dataset,
        valid_dataset=test_dataset,
        test_dataset=None,
        metrics_key="acc",
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        batch_forward_func=batch_forward_func,
        batch_cal_loss_func=batch_cal_loss_func,
        batch_metrics_func=batch_metrics_func,
        metrics_cal_func=metrics_cal_func,
        collate_fn=collator,
        device=dev,
        train_dataset_sampler=train_dataset_sampler,
        valid_dataset_sampler=test_dataset_sampler,
        valid_step=1,
        start_epoch=0,
        gradient_accumulate=gradient_accumulate,
        save_model=False
    )
    trainer.train()
    metrics.append(trainer.epoch_metrics[trainer.get_best_epoch()])
    print(metrics)
    print()
    print(sum(list(map(lambda x: x["acc"], metrics)))/len(metrics))
