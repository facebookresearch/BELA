import h5py
import json
from pathlib import Path
from tqdm import tqdm
import logging
from transformers import AutoTokenizer, AutoModel
import numpy as np
import hydra
from omegaconf import OmegaConf, DictConfig
import torch

from duck.common.utils import device

logger = logging.getLogger()

class CatalogueBuilder:
    def __init__(self,
        input_path,
        output_tok_ids_path,
        output_idx_path,
        output_repr_path,
        model="bert-large-uncased",
        label_key="wikipedia_title",
        text_key="text",
        batch_size=256,
        max_seq_length=256,
        compute_representations=False
    ):
        self.input_path = Path(input_path)
        self.output_tok_ids_path = Path(output_tok_ids_path)
        self.output_repr_path = Path(output_repr_path) if output_repr_path else None
        self.output_idx_path = Path(output_idx_path)
        self.label_key = label_key
        self.text_key = text_key
        self.batch_size = batch_size
        self.max_length = max_seq_length
        self.device = device()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.language_model = AutoModel.from_pretrained(model).to(self.device).eval()
        self.compute_representations = compute_representations

        self.input_data = self._read_input()
        self.token_ids, self.representations = self._process()
        self.index = list(self.input_data.keys())

    def _read_input(self):
        logger.info("Reading input data")
        with open(self.input_path, 'r') as f:
            data = [json.loads(line.strip()) for line in tqdm(f.readlines())]
            if len(data) == 1:
                # json format
                return data[0]
            # jsonl format
            return {
                record[self.label_key]: record
                for record in data
            }

    def _process(self):
        logger.info("Tokenizing")
        token_ids = []
        representations = []
        data = list(self.input_data.values())
        for i in tqdm(range(0, len(data), self.batch_size)):
            batch_raw = data[i:i + self.batch_size]
            batch = []
            for entry in batch_raw:
                label = entry[self.label_key]
                text = entry[self.text_key]
                if isinstance(text, list):
                    text = "".join(entry["text"][:10])
                description = f"{label} {self.tokenizer.sep_token} {text}"
                batch.append(description)
            tokens = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.max_length,
                return_tensors='pt'
            )
            input_ids = tokens.input_ids.to(self.device)
            attention_mask = tokens.attention_mask.to(self.device)
            if self.compute_representations:
                with torch.no_grad():
                    output = self.language_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    token_representations = output["last_hidden_state"]
                    cls_token = token_representations[:, 0, :]
                    representations.append(cls_token.cpu().numpy())
            token_ids.append(input_ids.cpu().numpy().astype(np.int32))
        token_ids = np.concatenate(token_ids, axis=0)
        if self.compute_representations:
            representations = np.concatenate(representations, axis=0)
        size_prefix = self.max_length * np.ones((token_ids.shape[0], 1), dtype=np.int32)
        token_ids = np.concatenate([size_prefix, token_ids], axis=1)
        if not self.compute_representations:
            representations = None
        return token_ids, representations
    
    def _save(self):
        logger.info("Saving")
        with h5py.File(self.output_tok_ids_path, "w") as f:
            f['data'] = self.token_ids
        with h5py.File(self.output_repr_path, "w") as f:
            f['data'] = self.representations
        with open(self.output_idx_path, "w") as f:
            f.writelines("\n".join(self.index))

    def build(self):
        self._save()


@hydra.main(config_path="../conf/preprocessing", config_name="catalogue")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    CatalogueBuilder(**config).build()


if __name__ == "__main__":
    main()
