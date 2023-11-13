import argparse
import json
from model import *
from data_preprocess import *
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import os
parser=argparse.ArgumentParser()
parser.add_argument("--main_device",type=int,default=2)
parser.add_argument("--device_ids",type=str,default="2,3")
parser.add_argument("--batch_size",type=int,default=6)
parser.add_argument("--num_workers",type=int,default=1)
parser.add_argument("--learning_rate",type=float,default=0.000001)
parser.add_argument("--epochs",type=int,default=50)
parser.add_argument("--mlm_name_or_path",type=str,default="hfl/chinese-roberta-wwm-ext")
parser.add_argument("--save_path",type=str,default="roberta")


if __name__=='__main__':
    args=parser.parse_args()
    """一些常规的设置"""
    dev=torch.device(args.main_device)
    device_ids=list(map(lambda x:int(x),args.device_ids.split(",")))
    batch_size=args.batch_size
    num_workers=args.num_workers
    learning_rate=args.learning_rate
    epochs=args.epochs
    mlm_type=args.mlm_name_or_path
    save_path=os.path.join('/data','zxy',args.save_path+'_saved')
    #print(save_path)

    train_dataset=[]
    test_dataset=[]
    with open("/home/zxy/coling/data/scm/train.json","r") as f:
        for line in f:
            if line:
                js=json.loads(line)
                train_dataset.append(Inputfeature(js['A'],js['B'],js['C'],js['label']))
    with open("/home/zxy/coling/data/scm/test.json","r") as f:
        for line in f:
            if line:
                js=json.loads(line)
                test_dataset.append(Inputfeature(js['A'],js['B'],js['C'],js['label']))

    tokenizer=BertTokenizer.from_pretrained(mlm_type)
    collator=Collator(tokenizer)
    batch_cal_loss_func=BatchCalLossFunc(1.0)
    model=BertBaselineModel(mlm_type)
    optimizer=get_optimizer(model,learning_rate)
    model=torch.nn.DataParallel(model,device_ids=device_ids)
    metrics=[]
    

    train_dataset_sampler=RandomSampler(train_dataset)
    test_dataset_sampler=SequentialSampler(test_dataset)
    
    trainer=Trainer(
        model=model,
        optimizer=optimizer,
        output_dir=save_path,
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
        start_epoch=0
    )
    trainer.train()
    metrics.append(trainer.epoch_metrics[trainer.get_best_epoch()])
    print(metrics)
    print()
    print(sum(list(map(lambda x:x["acc"],metrics)))/len(metrics))