from datasets import load_dataset
from torch.utils.data import Dataset


class OpusParacrawl(Dataset):
    def __init__(
        self,
        config,
        tokenizer,
        split 
    ):
        print(config, "opus_paracrawl")
        self.dataset = load_dataset("opus_paracrawl", lang1="en", lang2="uk")['train']
        if split == "train":
            self.dataset = self.dataset.select(list(range(0, 10_000)))
        else:
            self.dataset = self.dataset.select(list(range(10_000, 11_000)))
        self.tokenizer = tokenizer

    def __len__(self):
        return self.dataset.shape[0]

    def convert_to_features(self, example_batch):

        # Create prompt and tokenize contexts and questions

        input_ = example_batch["translation"]["en"]
        target_ = example_batch["translation"]["uk"]

        prompt = f"Translate from English to Ukrainian: {input_}\n---\nTranslated: "
        prompt_ids = self.tokenizer.encode(self.tokenizer.bos_token + prompt, add_special_tokens=False)
        label_ids = self.tokenizer.encode(target_ + self.tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt_ids + label_ids,
            "attention_mask": [1] * len(prompt_ids + label_ids),
            "labels": [-100] * len(prompt_ids) + label_ids
        }

        return sample

    def __getitem__(self, index):
        return self.convert_to_features(self.dataset[int(index)])
