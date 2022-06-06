import unittest

import torch
from bela.models.hf_encoder import HFEncoder
from bela.transforms.joint_el_transform import JointELTransform

class TestHFEncoder(unittest.TestCase):
    def test_xlmr_encoder(self):
        transform = JointELTransform()
        model = HFEncoder(model_path="xlm-roberta-base")

        model_inputs = transform(
            {
                "texts": [
                    [
                        "Some",
                        "simple",
                        "text",
                        "about",
                        "Real",
                        "Madrid",
                        "and",
                        "Barcelona",
                    ],
                    ["Hola", "amigos", "!"],
                    ["Cristiano", "Ronaldo", "juega", "en", "la", "Juventus"],
                ],
                "mention_offsets": [
                    [4, 7],
                    [1],
                    [0, 5],
                ],
                "mention_lengths": [
                    [2, 1],
                    [1],
                    [2, 1],
                ],
                "entities": [
                    [1, 2],
                    [3],
                    [102041, 267832],
                ],
            }
        )

        output = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
        )


if __name__ == '__main__':
    unittest.main()
