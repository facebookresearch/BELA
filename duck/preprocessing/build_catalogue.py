import h5py
import json
from pathlib import Path
from tqdm import tqdm
import logging
from transformers import AutoTokenizer
import numpy as np
import hydra
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger()

class CatalogueBuilder:
    def __init__(self,
        input_path,
        output_h5_path,
        output_idx_path,
        tokenizer="bert-large-uncased",
        label_key="wikipedia_title",
        text_key="text",
        batch_size=256,
        max_seq_length=256
    ):
        self.input_path = Path(input_path)
        self.output_h5_path = Path(output_h5_path)
        self.output_idx_path = Path(output_idx_path)
        self.label_key = label_key
        self.text_key = text_key
        self.batch_size = batch_size
        self.max_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.input_data = self._read_input()
        self.token_ids = self._tokenize()
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

    def _tokenize(self):
        logger.info("Tokenizing")
        result = []
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
                return_tensors='np'
            )
            input_ids = tokens.input_ids.astype(np.int32)
            result.append(input_ids)
        result = np.concatenate(result, axis=0)
        size_prefix = self.max_length * np.ones((result.shape[0], 1), dtype=np.int32)
        return np.concatenate([size_prefix, result], axis=1)
    
    def _save(self):
        logger.info("Saving")
        with h5py.File(self.output_h5_path, "w") as f:
            f['data'] = self.token_ids
        with open(self.output_idx_path, "w") as f:
            f.writelines("\n".join(self.index))

    def build(self):
        self._save()


@hydra.main(config_path="../conf/preprocessing", config_name="catalogue")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    config_dict = OmegaConf.to_container(config)
    CatalogueBuilder(**config_dict).build()


if __name__ == "__main__":
    main()
