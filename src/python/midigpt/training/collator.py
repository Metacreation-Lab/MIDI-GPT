from dataclasses import dataclass

import torch


@dataclass
class MidiGPTCollator:
    """Pads variable-length token sequences to the longest in the batch.

    Returns dict with keys: input_ids, attention_mask, labels.
    Labels are set to pad_value (-100) for padding positions so the loss
    function ignores them.
    """

    max_seq_len: int = 2048
    pad_value: int = -100

    def __call__(self, batch: list[dict]) -> dict:
        input_ids = [
            torch.tensor(item["input_ids"][: self.max_seq_len], dtype=torch.long) for item in batch
        ]
        max_len = max(t.size(0) for t in input_ids)

        padded_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels = torch.full((len(batch), max_len), self.pad_value, dtype=torch.long)

        for i, ids in enumerate(input_ids):
            seq_len = ids.size(0)
            padded_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = ids

        return {
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
