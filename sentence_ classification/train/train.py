#!/usr/local/bin/python
import subprocess

print('hello world!!');

subprocess.call('./cleancache.sh', shell=True)

# 모델 환경 설정
import torch
from ratsnlp.nlpbook.classification import ClassificationTrainArguments
args = ClassificationTrainArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_task_name="pair-classification",
    downstream_corpus_name="klue-nli",
    downstream_model_dir="/gdrive/My Drive/nlpbook/checkpoint-paircls",
    batch_size=32 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,
    max_seq_length=128,
    epochs=3,
    tpu_cores=0,
    seed=7,
)

# 랜덤 시드 고정
from ratsnlp import nlpbook
nlpbook.set_seed(args)

# 로거 설정
nlpbook.set_logger(args)

nlpbook.download_downstream_dataset(args)

# 토큰나이저 준비
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)

# 학습데이터 구축
from ratsnlp.nlpbook.paircls import KlueNLICorpus
from ratsnlp.nlpbook.classification import ClassificationDataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
corpus = KlueNLICorpus()
train_dataset = ClassificationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train",
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=RandomSampler(train_dataset, replacement=False),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)

# 테스트 데이터 구축축
val_dataset = ClassificationDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="test",
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler=SequentialSampler(val_dataset),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
    num_workers=args.cpu_workers,
)

# 모델 초기화
from transformers import BertConfig, BertForSequenceClassification
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=corpus.num_labels,
)

model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
)

# 학습 준비
from ratsnlp.nlpbook.classification import ClassificationTask
task = ClassificationTask(model, args)

trainer = nlpbook.get_trainer(args)

# 학습습
trainer.fit(
    task,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)