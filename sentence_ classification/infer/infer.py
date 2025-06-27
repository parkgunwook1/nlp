
from google.colab import drive
drive.mount('/gdrive', force_remount=True)


from ratsnlp.nlpbook.classification import ClassificationDeployArguments
args = ClassificationDeployArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_model_dir="/gdrive/My Drive/nlpbook/checkpoint-paircls",
    max_seq_length=64,
)

import torch
from transformers import BertConfig, BertForSequenceClassification
fine_tuned_model_ckpt = torch.load(
    args.downstream_model_checkpoint_fpath,
    map_location=torch.device("cpu")
)
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
)
model = BertForSequenceClassification(pretrained_model_config)
model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
model.eval()


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)


def inference_fn(premise, hypothesis):
    inputs = tokenizer(
        [(premise, hypothesis)],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    with torch.no_grad():
        outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
        prob = outputs.logits.softmax(dim=1)
        entailment_prob = round(prob[0][0].item(), 2)
        contradiction_prob = round(prob[0][1].item(), 2)
        neutral_prob = round(prob[0][2].item(), 2)
        if torch.argmax(prob) == 0:
            pred = "참 (entailment)"
        elif torch.argmax(prob) == 1:
            pred = "거짓 (contradiction)"
        else:
            pred = "중립 (neutral)"
    return {
        'premise': premise,
        'hypothesis': hypothesis,
        'prediction': pred,
        'entailment_data': f"참 {entailment_prob}",
        'contradiction_data': f"거짓 {contradiction_prob}",
        'neutral_data': f"중립 {neutral_prob}",
        'entailment_width': f"{entailment_prob * 100}%",
        'contradiction_width': f"{contradiction_prob * 100}%",
        'neutral_width': f"{neutral_prob * 100}%",
    }


from ratsnlp.nlpbook.paircls import get_web_service_app
app = get_web_service_app(inference_fn)
app.run()