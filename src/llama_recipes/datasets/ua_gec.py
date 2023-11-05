from torch.utils.data import Dataset, Subset, ConcatDataset
from ua_gec import Corpus


class UaGec(Dataset):
    def __init__(
        self,
        config,
        tokenizer,
        split 
    ):
        print(config, "ua_gec")
        self.train, self.dev = range(0, 1650), range(1650, 1706)
        self.fluency = Corpus(partition="train", annotation_layer="gec-fluency").get_documents()
        self.gec = Corpus(partition="train", annotation_layer="gec-only").get_documents()
        if split == "train":
            self.dataset = ConcatDataset([Subset(self.fluency, self.train)])
        else:
            self.dataset = ConcatDataset([Subset(self.fluency, self.dev)])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, doc):
        prompt = f"Виправ граматику: {doc.source}\n---\nВиправлення: "
        prompt_ids = self.tokenizer.encode(self.tokenizer.bos_token + prompt, add_special_tokens=False)
        label_ids = self.tokenizer.encode(doc.target + self.tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt_ids + label_ids,
            "attention_mask": [1] * len(prompt_ids + label_ids),
            "labels": [-100] * len(prompt_ids) + label_ids
        }
        return sample

    def __getitem__(self, index):
        return self.convert_to_features(self.dataset[int(index)])
