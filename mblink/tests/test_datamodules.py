import unittest
import os
import tempfile
import random
import torch

import h5py
import numpy as np
import torch
from mblink.datamodule.blink_datamodule import (
    ElBlinkDataset,
    ElMatchaDataset,
    ElBiEncoderDataModule,
    EntityCatalogue,
    MultilangEntityCatalogue,
)
from mblink.transforms.blink_transform import (
    BlinkTransform,
)
from mblink.utils.utils import assert_equal_tensor_dict


class TestBiEncoderELDataModule(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        torch.manual_seed(0)
        self.base_dir = os.path.join(os.path.dirname(__file__), "data")
        self.data_path = os.path.join(self.base_dir, "el_matcha.jsonl")
        self.ent_catalogue_path = os.path.join(self.base_dir, "el_catalogue.h5")
        self.ent_catalogue_idx_path = os.path.join(self.base_dir, "el_catalogue.idx")

        self.transform = BlinkTransform(
            model_path="bert-large-uncased",
            max_mention_len=12,
            max_entity_len=64,
            add_eos_bos_to_entity=True,
        )

        self.tokens = {
            "London": [
                2414,
                2003,
                1996,
                3007,
                1998,
                2922,
                2103,
                1997,
                2563,
                1998,
                1996,
                2142,
                2983,
            ],
            "Chelsea F.C.": [
                9295,
                2374,
                2252,
                2003,
                2019,
                2394,
                2658,
                2374,
                2252,
                2241,
                1999,
                21703,
                1010,
                2414,
                1012,
            ],
        }

    def test_ent_catalogue(self):
        ent_catalogue = EntityCatalogue(
            self.ent_catalogue_path,
            self.ent_catalogue_idx_path,
        )
        self.assertIn("London", ent_catalogue)
        self.assertIn("Chelsea F.C.", ent_catalogue)
        self.assertNotIn("Moscow", ent_catalogue)

        idx, data = ent_catalogue["London"]
        self.assertEqual(idx, 0)
        self.assertSequenceEqual(data, self.tokens["London"])

        idx, data = ent_catalogue["Chelsea F.C."]
        self.assertEqual(idx, 1)
        self.assertSequenceEqual(data, self.tokens["Chelsea F.C."])

    def test_el_matcha_dataset(self):
        ent_catalogue = EntityCatalogue(
            self.ent_catalogue_path,
            self.ent_catalogue_idx_path,
        )

        ds = ElMatchaDataset(path=self.data_path, ent_catalogue=ent_catalogue)

        self.assertEqual(len(ds), 3)
        self.assertEqual(
            ds[0],
            {
                "context_left": "",
                "mention": "Chelsea Football Club",
                "context_right": "is an English professional football club based in Fulham London",
                "entity_id": "Chelsea F.C.",
                "entity_index": 1,
                "entity_tokens": self.tokens["Chelsea F.C."],
            },
        )

        self.assertEqual(
            ds[1],
            {
                "context_left": "Chelsea Football Club is an English professional football club based in Fulham",
                "mention": "London",
                "context_right": "",
                "entity_id": "London",
                "entity_index": 0,
                "entity_tokens": self.tokens["London"],
            },
        )

        self.assertEqual(
            ds[2],
            {
                "context_left": "In",
                "mention": "London",
                "context_right": "the capital of England and the United Kingdom",
                "entity_id": "London",
                "entity_index": 0,
                "entity_tokens": self.tokens["London"],
            },
        )

    def test_el_bi_encoder_data_module(self):
        dm = ElBiEncoderDataModule(
            transform=self.transform,
            train_path=self.data_path,
            val_path=self.data_path,
            test_path=self.data_path,
            ent_catalogue_path=self.ent_catalogue_path,
            ent_catalogue_idx_path=self.ent_catalogue_idx_path,
            batch_size=2,
        )

        batches = list(dm.train_dataloader())
        self.assertEqual(len(batches), 2)

        expected_batches = [
            {
                "mentions": {
                    "input_ids": torch.tensor(
                        [
                            [
                                101,
                                1,
                                9295,
                                2374,
                                2252,
                                2,
                                2003,
                                2019,
                                2394,
                                2658,
                                2374,
                                102,
                            ],
                            [
                                101,
                                2394,
                                2658,
                                2374,
                                2252,
                                2241,
                                1999,
                                21703,
                                1,
                                2414,
                                2,
                                102,
                            ],
                        ]
                    ),
                    "attention_mask": torch.tensor(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                },
                "entities": {
                    "input_ids": torch.tensor(
                        [
                            [
                                101,
                                9295,
                                2374,
                                2252,
                                2003,
                                2019,
                                2394,
                                2658,
                                2374,
                                2252,
                                2241,
                                1999,
                                21703,
                                1010,
                                2414,
                                1012,
                                102,
                            ],
                            [
                                101,
                                2414,
                                2003,
                                1996,
                                3007,
                                1998,
                                2922,
                                2103,
                                1997,
                                2563,
                                1998,
                                1996,
                                2142,
                                2983,
                                102,
                                0,
                                0,
                            ],
                        ]
                    ),
                    "attention_mask": torch.tensor(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        ]
                    ),
                },
                "entity_ids": torch.tensor([1, 0]),
                "targets": torch.tensor([0, 1]),
                "entity_tensor_mask": torch.tensor([1, 1]),
            },
            {
                "mentions": {
                    "input_ids": torch.tensor(
                        [
                            [
                                101,
                                1999,
                                1,
                                2414,
                                2,
                                1996,
                                3007,
                                1997,
                                2563,
                                1998,
                                1996,
                                102,
                            ]
                        ]
                    ),
                    "attention_mask": torch.tensor(
                        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
                    ),
                },
                "entities": {
                    "input_ids": torch.tensor(
                        [
                            [
                                101,
                                2414,
                                2003,
                                1996,
                                3007,
                                1998,
                                2922,
                                2103,
                                1997,
                                2563,
                                1998,
                                1996,
                                2142,
                                2983,
                                102,
                            ]
                        ]
                    ),
                    "attention_mask": torch.tensor(
                        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
                    ),
                },
                "entity_ids": torch.tensor([0]),
                "targets": torch.tensor([0]),
                "entity_tensor_mask": torch.tensor([1]),
            },
        ]

        for result, expected in zip(batches, expected_batches):
            assert_equal_tensor_dict(self, result, expected)


class TestBiEncoderELDataModuleWithXlmrTransform(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        torch.manual_seed(0)
        self.base_dir = os.path.join(os.path.dirname(__file__), "data")
        self.data_path = os.path.join(self.base_dir, "el_matcha.jsonl")
        self.ent_catalogue_path = os.path.join(self.base_dir, "el_xlmr_catalogue.h5")
        self.ent_catalogue_idx_path = os.path.join(self.base_dir, "el_catalogue.idx")

        self.transform = BlinkTransform(
            model_path="xlm-roberta-base",
            mention_start_token=-2,
            mention_end_token=-3,
            max_mention_len=12,
            max_entity_len=32,
        )

    def test_el_bi_encoder_data_module_with_xlmr_transform(self):
        dm = ElBiEncoderDataModule(
            transform=self.transform,
            train_path=self.data_path,
            val_path=self.data_path,
            test_path=self.data_path,
            ent_catalogue_path=self.ent_catalogue_path,
            ent_catalogue_idx_path=self.ent_catalogue_idx_path,
            batch_size=2,
        )

        batches = list(dm.train_dataloader())
        self.assertEqual(len(batches), 2)

        expected_batches = [
            {
                "mentions": {
                    "input_ids": torch.tensor(
                        [
                            [
                                0,
                                250000,
                                44517,
                                98809,
                                7687,
                                249999,
                                83,
                                142,
                                14941,
                                23182,
                                101740,
                                2,
                            ],
                            [
                                0,
                                23182,
                                101740,
                                11938,
                                35509,
                                23,
                                88437,
                                3915,
                                250000,
                                9020,
                                249999,
                                2,
                            ],
                        ]
                    ),
                    "attention_mask": torch.tensor(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                },
                "entities": {
                    "input_ids": torch.tensor(
                        [
                            [
                                0,
                                44517,
                                563,
                                5,
                                441,
                                5,
                                250000,
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
                                4,
                                9020,
                                5,
                                215624,
                                297,
                                23,
                                66007,
                                4,
                                70,
                                11938,
                                98438,
                                2,
                            ],
                            [
                                0,
                                9020,
                                250000,
                                9020,
                                83,
                                70,
                                10323,
                                136,
                                142105,
                                26349,
                                111,
                                30715,
                                136,
                                70,
                                14098,
                                117604,
                                5,
                                581,
                                26349,
                                9157,
                                7,
                                98,
                                70,
                                32547,
                                99321,
                                90,
                                23,
                                70,
                                127067,
                                9,
                                13,
                                2,
                            ],
                        ]
                    ),
                    "attention_mask": torch.tensor(
                        [
                            [
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                            ],
                            [
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                            ],
                        ]
                    ),
                },
                "entity_ids": torch.tensor([1, 0]),
                "targets": torch.tensor([0, 1]),
                "entity_tensor_mask": torch.tensor([1, 1]),
            },
            {
                "mentions": {
                    "input_ids": torch.tensor(
                        [
                            [
                                0,
                                360,
                                250000,
                                9020,
                                249999,
                                70,
                                10323,
                                111,
                                30715,
                                136,
                                70,
                                2,
                            ]
                        ]
                    ),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
                },
                "entities": {
                    "input_ids": torch.tensor(
                        [
                            [
                                0,
                                9020,
                                250000,
                                9020,
                                83,
                                70,
                                10323,
                                136,
                                142105,
                                26349,
                                111,
                                30715,
                                136,
                                70,
                                14098,
                                117604,
                                5,
                                581,
                                26349,
                                9157,
                                7,
                                98,
                                70,
                                32547,
                                99321,
                                90,
                                23,
                                70,
                                127067,
                                9,
                                13,
                                2,
                            ]
                        ]
                    ),
                    "attention_mask": torch.tensor(
                        [
                            [
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                            ]
                        ]
                    ),
                },
                "entity_ids": torch.tensor([0]),
                "targets": torch.tensor([0]),
                "entity_tensor_mask": torch.tensor([1]),
            },
        ]

        for result, expected in zip(batches, expected_batches):
            assert_equal_tensor_dict(self, result, expected)

    def test_el_bi_encoder_data_module_with_hard_negatives_with_xlmr_transform(self):
        dm = ElBiEncoderDataModule(
            transform=self.transform,
            train_path=self.data_path,
            val_path=self.data_path,
            test_path=self.data_path,
            ent_catalogue_path=self.ent_catalogue_path,
            ent_catalogue_idx_path=self.ent_catalogue_idx_path,
            batch_size=2,
            negatives=True,
            max_negative_entities_in_batch=5,
        )

        batches = list(dm.train_dataloader())
        self.assertEqual(len(batches), 2)

        expected_batches = [
            {
                "mentions": {
                    "input_ids": torch.tensor(
                        [
                            [
                                0,
                                250000,
                                44517,
                                98809,
                                7687,
                                249999,
                                83,
                                142,
                                14941,
                                23182,
                                101740,
                                2,
                            ],
                            [
                                0,
                                23182,
                                101740,
                                11938,
                                35509,
                                23,
                                88437,
                                3915,
                                250000,
                                9020,
                                249999,
                                2,
                            ],
                        ]
                    ),
                    "attention_mask": torch.tensor(
                        [
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ]
                    ),
                },
                "entities": {
                    "input_ids": torch.tensor(
                        [
                            [
                                0,
                                44517,
                                563,
                                5,
                                441,
                                5,
                                250000,
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
                                4,
                                9020,
                                5,
                                215624,
                                297,
                                23,
                                66007,
                                4,
                                70,
                                11938,
                                98438,
                                2,
                            ],
                            [
                                0,
                                9020,
                                250000,
                                9020,
                                83,
                                70,
                                10323,
                                136,
                                142105,
                                26349,
                                111,
                                30715,
                                136,
                                70,
                                14098,
                                117604,
                                5,
                                581,
                                26349,
                                9157,
                                7,
                                98,
                                70,
                                32547,
                                99321,
                                90,
                                23,
                                70,
                                127067,
                                9,
                                13,
                                2,
                            ],
                            [0, 2] + [1] * 30,
                            [0, 2] + [1] * 30,
                            [0, 2] + [1] * 30,
                            [0, 2] + [1] * 30,
                            [0, 2] + [1] * 30,
                        ]
                    ),
                    "attention_mask": torch.tensor(
                        [
                            [1] * 32,
                            [1] * 32,
                            [1, 1] + [0] * 30,
                            [1, 1] + [0] * 30,
                            [1, 1] + [0] * 30,
                            [1, 1] + [0] * 30,
                            [1, 1] + [0] * 30,
                        ]
                    ),
                },
                "entity_ids": torch.tensor([1, 0, 0, 0, 0, 0, 0]),
                "targets": torch.tensor([0, 1]),
                "entity_tensor_mask": torch.tensor([1, 1, 0, 0, 0, 0, 0]),
            },
            {
                "mentions": {
                    "input_ids": torch.tensor(
                        [
                            [
                                0,
                                360,
                                250000,
                                9020,
                                249999,
                                70,
                                10323,
                                111,
                                30715,
                                136,
                                70,
                                2,
                            ]
                        ]
                    ),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
                },
                "entities": {
                    "input_ids": torch.tensor(
                        [
                            [
                                0,
                                9020,
                                250000,
                                9020,
                                83,
                                70,
                                10323,
                                136,
                                142105,
                                26349,
                                111,
                                30715,
                                136,
                                70,
                                14098,
                                117604,
                                5,
                                581,
                                26349,
                                9157,
                                7,
                                98,
                                70,
                                32547,
                                99321,
                                90,
                                23,
                                70,
                                127067,
                                9,
                                13,
                                2,
                            ],
                            [
                                0,
                                44517,
                                563,
                                5,
                                441,
                                5,
                                250000,
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
                                4,
                                9020,
                                5,
                                215624,
                                297,
                                23,
                                66007,
                                4,
                                70,
                                11938,
                                98438,
                                2,
                            ],
                            [0, 2] + [1] * 30,
                            [0, 2] + [1] * 30,
                            [0, 2] + [1] * 30,
                            [0, 2] + [1] * 30,
                        ]
                    ),
                    "attention_mask": torch.tensor(
                        [
                            [1] * 32,
                            [1] * 32,
                            [1, 1] + [0] * 30,
                            [1, 1] + [0] * 30,
                            [1, 1] + [0] * 30,
                            [1, 1] + [0] * 30,
                        ]
                    ),
                },
                "entity_ids": torch.tensor([0, 1, 0, 0, 0, 0]),
                "targets": torch.tensor([0]),
                "entity_tensor_mask": torch.tensor([1, 1, 0, 0, 0, 0]),
            },
        ]

        for result, expected in zip(batches, expected_batches):
            assert_equal_tensor_dict(self, result, expected)


class TestMultilangELDataModule(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        torch.manual_seed(0)
        self.base_dir = os.path.join(os.path.dirname(__file__), "data")
        self.data_path = os.path.join(self.base_dir, "el_blink.jsonl")
        self.ent_catalogue_idx_path = os.path.join(
            self.base_dir, "el_multi_catalogue.idx"
        )
        fid, self.ent_catalogue_path = tempfile.mkstemp()
        os.close(fid)
        self._create_ent_data(self.ent_catalogue_path)

    def tearDown(self):
        if os.path.isfile(self.ent_catalogue_path):
            os.remove(self.ent_catalogue_path)

    @staticmethod
    def _create_ent_data(file_name):
        with h5py.File(file_name, "w") as fd:
            fd["en"] = np.array(
                [
                    [3, 101, 25550, 102, 0, 0],
                    [3, 101, 16765, 102, 0, 0],
                    [5, 101, 12109, 10104, 14822, 102],
                    [3, 101, 10829, 102, 0, 0],
                ]
            )
            fd["pt"] = np.array(
                [
                    [3, 101, 12264, 102, 0, 0],
                    [5, 101, 14734, 47630, 27171, 102],
                ]
            )
            fd["ru"] = np.array([[5, 101, 59049, 118, 11323, 102]])

    def test_multilang_ent_catalogue(self):
        ent_catalogue = MultilangEntityCatalogue(
            self.ent_catalogue_path,
            self.ent_catalogue_idx_path,
        )

        self.assertIn("Q5146", ent_catalogue)
        self.assertIn("Q155", ent_catalogue)
        self.assertIn("Q8678", ent_catalogue)
        self.assertIn("Q84", ent_catalogue)
        self.assertNotIn("London", ent_catalogue)

        idx0, data = ent_catalogue["Q5146"]
        self.assertSequenceEqual(data, [101, 14734, 47630, 27171, 102])

        idx1, data = ent_catalogue["Q155"]
        self.assertSequenceEqual(data, [101, 16765, 102])

        idx2, data = ent_catalogue["Q8678"]
        self.assertSequenceEqual(data, [101, 59049, 118, 11323, 102])

        idx3, data = ent_catalogue["Q84"]
        self.assertSequenceEqual(data, [101, 10829, 102])

        # assert all keys have unique idx numbers
        self.assertEqual(sorted([idx0, idx1, idx2, idx3]), [0, 1, 2, 3])

    def test_el_blink_dataset(self):
        ent_catalogue = MultilangEntityCatalogue(
            self.ent_catalogue_path,
            self.ent_catalogue_idx_path,
        )

        ds = ElBlinkDataset(path=self.data_path, ent_catalogue=ent_catalogue)

        self.assertEqual(len(ds), 7)
        self.assertEqual(
            ds[0],
            {
                "context_left": "Guanabara K\u00f6rfezi (",
                "mention": "Portekizce",
                "context_right": ": ve de Rio de Janeiro eyaletinde.",
                "entity_id": "Q5146",
                "entity_index": ent_catalogue["Q5146"][0],
                "entity_tokens": ent_catalogue["Q5146"][1],
            },
        )

        self.assertEqual(
            ds[6],
            {
                "context_left": "Serpenti Galerisi (\u0130ngilizce: Serpentine Gallery),",
                "mention": "Londra",
                "context_right": "\u015fehrindeki Hyde Park\u2019\u0131n bir par\u00e7as\u0131 olan Kensington Gardens.",
                "entity_id": "Q84",
                "entity_index": ent_catalogue["Q84"][0],
                "entity_tokens": ent_catalogue["Q84"][1],
            },
        )

    def test_el_multilang_datamodule(self):
        transform = BlinkTransform(
            model_path="xlm-roberta-base",
            mention_start_token=-2,
            mention_end_token=-3,
            max_mention_len=12,
            max_entity_len=32,
        )

        dm = ElBiEncoderDataModule(
            transform=transform,
            train_path=self.data_path,
            val_path=self.data_path,
            test_path=self.data_path,
            ent_catalogue_path=self.ent_catalogue_path,
            ent_catalogue_idx_path=self.ent_catalogue_idx_path,
            dataset_type="blink",
            ent_catalogue_type="multi",
            batch_size=2,
            mention_start_token=1,
            mention_end_token=2,
            ent_sep_token=3,
            mention_context_length=12,
            separate_segments=True,
        )

        batches = list(dm.train_dataloader())
        self.assertEqual(len(batches), 4)




if __name__ == '__main__':
    unittest.main()
