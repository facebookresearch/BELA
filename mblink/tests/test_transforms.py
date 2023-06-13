# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from bela.transforms.joint_el_transform import JointELTransform

class TestJointELXlmrTransforms(unittest.TestCase):
    def test_blink_mention_xlmr_transform(self):
        transform = JointELTransform()
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

        expected_model_inputs = {
            "input_ids": torch.tensor(
                [
                    [0, 31384, 8781, 7986, 1672, 5120, 8884, 136, 5755, 2],
                    [0, 47958, 19715, 711, 2, 1, 1, 1, 1, 1],
                    [0, 96085, 43340, 1129, 2765, 22, 21, 65526, 2, 1],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                ]
            ),
            "mention_offsets": torch.tensor([[5, 8], [2, 0], [1, 7]]),
            "mention_lengths": torch.tensor([[2, 1], [1, 0], [2, 1]]),
            "entities": torch.tensor([[1, 2], [3, 0], [102041, 267832]]),
            "tokens_mapping": torch.tensor(
                [
                    [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]],
                    [
                        [1, 2],
                        [2, 3],
                        [3, 4],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                    ],
                    [
                        [1, 2],
                        [2, 3],
                        [3, 5],
                        [5, 6],
                        [6, 7],
                        [7, 8],
                        [0, 1],
                        [0, 1],
                    ],
                ]
            ),
        }

        for key, value in expected_model_inputs.items():
            self.assertTrue(
                torch.all(model_inputs[key].eq(value)), f"{key} not equal"
            )

if __name__ == '__main__':
    unittest.main()
