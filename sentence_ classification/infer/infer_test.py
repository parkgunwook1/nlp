#!/usr/bin/env python
# -*- coding: utf-8 -*-

print("욕설 탐지 모델 인퍼런스 시작")

import torch
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

# 모델 및 토크나이저 설정
MODEL_DIR = "/home/pkw85428/model/ckpt/checkpoint-72"
PRETRAINED_MODEL_NAME = "beomi/kcbert-base"
MAX_SEQ_LENGTH = 64

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, do_lower_case=False)

config = BertConfig.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR, config=config)
model.eval()

# 예측 함수
def inference_fn(premise: str, hypothesis: str):
    inputs = tokenizer(
        [(premise, hypothesis)],
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        truncation=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

        pred_idx = torch.argmax(probs, dim=1).item()
        label_map = {
            0: " 욕설",
            1: " 정상",
            2: " 중립"
        }
        pred = label_map.get(pred_idx, "알 수 없음")

        return {
            "premise": premise,
            "hypothesis": hypothesis,
            "판단": pred,
            "확률": {
                "욕설": round(probs[0][0].item(), 4),
                "정상": round(probs[0][1].item(), 4),
                "중립": round(probs[0][2].item(), 4),
            }
        }

# CLI 입력 루프
print("\n💬 예시: '이 문장은 욕설이다' + 테스트 문장")
while True:
    try:
        hypo = input("\n[입력] 판별할 문장을 입력하세요 (종료: Ctrl+C): ")
        result = inference_fn("이 문장은 욕설이다", hypo)

        print("\n[판별 결과]")
        print(f"문장: {result['hypothesis']}")
        print(f"판단: {result['판단']}")
        print("확률:")
        for k, v in result["확률"].items():
            print(f"  - {k}: {v}")
    except KeyboardInterrupt:
        print("\n 종료합니다.")
        break
