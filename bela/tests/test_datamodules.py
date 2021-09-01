import unittest
import os

import torch
from bela.transforms.joint_el_transform import JointELTransform
from bela.datamodule.joint_el_datamodule import JointELDataModule


def assert_equal_tensor_dict(test_case, result, expected):
    """
    Compare tensors/values in the dict and assert if they are not equal.
    The dict could countain multiple levels of nesting.
    """
    for key, value in expected.items():
        if isinstance(value, dict):
            assert_equal_tensor_dict(test_case, result[key], value)
        else:
            if isinstance(value, torch.Tensor):
                test_case.assertTrue(
                    torch.equal(result[key], value), f"{key} is not equal"
                )
            else:
                test_case.assertEqual(result[key], value, f"{key} is not equal")


class TestJointELDataModule(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.join(os.path.dirname(__file__), "data")
        self.data_path = os.path.join(self.base_dir, "el_matcha_joint.jsonl")
        self.ent_catalogue_idx_path = os.path.join(self.base_dir, "el_catalogue.idx")

        self.transform = JointELTransform()

    def test_joint_el_datamodule_with_saliency_scores(self):
        dm = JointELDataModule(
            transform=self.transform,
            train_path=self.data_path,
            val_path=self.data_path,
            test_path=self.data_path,
            ent_catalogue_idx_path=self.ent_catalogue_idx_path,
            batch_size=2,
        )

        batches = list(dm.train_dataloader())
        self.assertEqual(len(batches), 1)

        expected_batches = [
            {
                "input_ids": torch.tensor(
                    [
                        [
                            0,
                            44517,
                            98809,
                            7687,
                            83,
                            142,
                            14941,
                            23182,
                            101740,
                            11938,
                            35509,
                            23,
                            88437,
                            3915,
                            9020,
                            2,
                        ],
                        [
                            0,
                            360,
                            9020,
                            70,
                            10323,
                            111,
                            30715,
                            136,
                            70,
                            14098,
                            117604,
                            2,
                            1,
                            1,
                            1,
                            1,
                        ],
                    ]
                ),
                "attention_mask": torch.tensor(
                    [
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    ]
                ),
                "mention_offsets": torch.tensor([[1, 14], [2, 0]]),
                "mention_lengths": torch.tensor([[3, 1], [1, 0]]),
                "entities": torch.tensor([[1, 0], [0, 0]]),
                "tokens_mapping": torch.tensor(
                    [
                        [
                            [1, 2],
                            [2, 3],
                            [3, 4],
                            [4, 5],
                            [5, 6],
                            [6, 7],
                            [7, 8],
                            [8, 9],
                            [9, 10],
                            [10, 11],
                            [11, 12],
                            [12, 14],
                            [14, 15],
                        ],
                        [
                            [1, 2],
                            [2, 3],
                            [3, 4],
                            [4, 5],
                            [5, 6],
                            [6, 7],
                            [7, 8],
                            [8, 9],
                            [9, 10],
                            [10, 11],
                            [0, 1],
                            [0, 1],
                            [0, 1],
                        ],
                    ]
                ),
                "salient_entities": [{1}, {0}],
            }
        ]

        for result, expected in zip(batches, expected_batches):
            assert_equal_tensor_dict(self, result, expected)



if __name__ == '__main__':
    unittest.main()
