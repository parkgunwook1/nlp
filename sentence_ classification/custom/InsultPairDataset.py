# custom_dataset.py
from torch.utils.data import Dataset
import csv

class InsultPairDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.samples.append({
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                    "label": int(row["label"]),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoded = self.tokenizer.encode_plus(
            sample["premise"],
            sample["hypothesis"],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "token_type_ids": encoded["token_type_ids"].squeeze(0),
            "labels": sample["label"]
        }
