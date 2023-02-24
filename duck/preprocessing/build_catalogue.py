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

from duck.common.utils import device, load_pkl

logger = logging.getLogger()

class CatalogueBuilder:
    def __init__(
        self,
        input_path,
        output_tok_ids_path,
        output_idx_path,
        output_repr_path=None,
        kb_path=None,
        model="bert-large-uncased",
        label_key="wikipedia_title",
        text_key="text",
        batch_size=256,
        max_seq_length=256
    ):
        self.input_path = Path(input_path)
        self.output_tok_ids_path = Path(output_tok_ids_path)
        self.output_repr_path = Path(output_repr_path) if output_repr_path else None
        self.output_idx_path = Path(output_idx_path)
        self.kb_path = Path(kb_path) if kb_path else None
        self.label_key = label_key
        self.text_key = text_key
        self.batch_size = batch_size
        self.max_length = max_seq_length
        self.device = device()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.language_model = AutoModel.from_pretrained(model).to(self.device).eval()
        self.compute_representations = output_repr_path is not None

        self.input_data = self._read_input()
        self.kb = self._read_kb()
        self.index, self.token_ids, self.representations = self._process()
    
    def _read_input(self):
        logger.info(f"Reading input data {str(self.input_path)}")
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
    
    def _read_kb(self):
        logger.info(f"Reading KB: {str(self.kb_path)}")
        if self.kb_path is None:
            return None
        kb = load_pkl(self.kb_path)
        in_kb_entries = [entry for entry in self.input_data if entry in kb]
        coverage = float(len(in_kb_entries)) / len(self.input_data)
        logger.info(f"KB coverage: {coverage:.4f}")
        return {k: sorted(list(v))[0] for k, v in kb.items()}
                
    def _process(self):
        logger.info("Tokenizing")
        token_ids = []
        representations = []
        index = []
        data = list(self.input_data.values())
        data_keys = list(self.input_data.keys())
        for i in tqdm(range(0, len(data), self.batch_size)):
            batch_raw = data[i:i + self.batch_size]
            batch_keys = data_keys[i: i + self.batch_size]
            batch = []
            for j, entry in enumerate(batch_raw):
                label = entry[self.label_key]
                if self.kb is None or label in self.kb:
                    text = entry[self.text_key]
                    if isinstance(text, list):
                        text = "".join(entry["text"][:10])
                    description = f"{label} {self.tokenizer.sep_token} {text}"
                    batch.append(description)
                    index_id = batch_keys[j] if self.kb is None else self.kb[label]
                    index.append(index_id)
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
        assert len(index) == token_ids.shape[0]
        return index, token_ids, representations
    
    def _save(self):
        logger.info("Saving")
        with h5py.File(self.output_tok_ids_path, "w") as f:
            f['data'] = self.token_ids
        if self.output_repr_path is not None:
            with h5py.File(self.output_repr_path, "w") as f:
                f['data'] = self.representations
        with open(self.output_idx_path, "w") as f:
            f.writelines("\n".join(self.index))

    def build(self):
        self._save()


@hydra.main(config_path="../conf/preprocessing", config_name="catalogue", version_base=None)
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    CatalogueBuilder(**config).build()


if __name__ == "__main__":
    main()
