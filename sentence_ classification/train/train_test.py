#!/usr/local/bin/python
import torch
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from custom_dataset import InsultPairDataset
from sklearn.metrics import accuracy_score, f1_score

print("문서 쌍 욕설 탐지 모델 학습 시작")

# 파라미터 설정
PRETRAINED_MODEL_NAME = "beomi/kcbert-base"
MODEL_DIR = "/home/pkw85428/model/ckpt"
DATA_PATH = "./insult_pair_dataset.txt"
VAL_DATA_PATH = './val_dataset.txt'  # 검증 데이터셋 경로 (예시로 동일한 데이터 사용)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average='macro')
    }

# 토크나이저
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# 데이터셋
train_dataset = InsultPairDataset(DATA_PATH, tokenizer)
val_dataset = InsultPairDataset(VAL_DATA_PATH, tokenizer)  # 검증 데이터셋

# 모델 구성
config = BertConfig.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=3)
model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, config=config)

# 학습 설정
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    # evaluation_strategy="epoch",
    # save_strategy="epoch",
    # load_best_model_at_end=True,
    logging_dir='./logs'
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 학습 시작
trainer.train()

print("학습 완료. 모델 저장 위치:", MODEL_DIR)


metrics = trainer.evaluate()
print("검증 결과:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
