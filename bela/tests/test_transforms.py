# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from bela.transforms.joint_el_transform import JointELTransform, JointELXlmrRawTextTransform


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
                    [[1, 2], [2, 3], [3, 4], [4, 5], [
                        5, 6], [6, 7], [7, 8], [8, 9]],
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
                torch.all(model_inputs[key].eq(value)), f"{key} not equal")

    def test_joint_el_raw_text_xlmr_transform(self):
        transform = JointELXlmrRawTextTransform()

        model_inputs = transform(
            {
                "texts": [
                    "Some simple text about Real Madrid and Barcelona",
                    "Cristiano Ronaldo juega en la Juventus",
                    "Hola amigos!",
                    "   Hola   amigos!   ",  # test extra spaces
                ],
                "mention_offsets": [
                    [23, 39],
                    [0, 30],
                    [5],
                    [10],
                ],
                "mention_lengths": [
                    [11, 9],
                    [17, 8],
                    [6],
                    [6],
                ],
                "entities": [
                    [1, 2],
                    [102041, 267832],
                    [3],
                    [3],
                ],
            }
        )

        expected_model_inputs = {
            "input_ids": torch.tensor(
                [
                    [0, 31384, 8781, 7986, 1672, 5120, 8884, 136, 5755, 2],
                    [0, 96085, 43340, 1129, 2765, 22, 21, 65526, 2, 1],
                    [0, 47958, 19715, 38, 2, 1, 1, 1, 1, 1],
                    # Whitespaces are ignored
                    [0, 47958, 19715, 38, 2, 1, 1, 1, 1, 1],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                ]
            ),
            "mention_offsets": torch.tensor([[5, 8], [1, 7], [2, 0], [2, 0]]),
            "mention_lengths": torch.tensor([[2, 1], [2, 1], [1, 0], [1, 0]]),
            "entities": torch.tensor([[1, 2], [102041, 267832], [3, 0], [3, 0]]),
            "tokens_mapping": torch.tensor(
                [
                    [[1, 2], [2, 3], [3, 4], [4, 5], [
                        5, 6], [6, 7], [7, 8], [8, 9]],
                    [
                        [1, 2],
                        [2, 3],
                        [3, 4],
                        [4, 5],
                        [5, 6],
                        [6, 7],
                        [7, 8],
                        [0, 1],
                    ],
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
                        [3, 4],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                    ],
                ]
            ),
            "sp_tokens_boundaries": torch.tensor(
                [
                    [
                        [0, 4],
                        [4, 11],
                        [11, 16],
                        [16, 22],
                        [22, 27],
                        [27, 34],
                        [34, 38],
                        [38, 48],
                    ],
                    [
                        [0, 9],
                        [9, 17],
                        [17, 20],
                        [20, 23],
                        [23, 26],
                        [26, 29],
                        [29, 38],
                        [0, 1],
                    ],
                    [[0, 4], [4, 11], [11, 12], [0, 1],
                        [0, 1], [0, 1], [0, 1], [0, 1]],
                    [[0, 4+3], [4+3, 11+5], [11+5, 12+5], [0, 1],
                        [0, 1], [0, 1], [0, 1], [0, 1]],  # Add whitespaces
                ]
            ),
        }

        for key, value in expected_model_inputs.items():
            self.assertTrue(torch.all(model_inputs[key].eq(
                value)), f"{key} not equal: {model_inputs[key]=} != {value=}")

    def test_joint_el_raw_text_xlmr_transform_2(self):
        examples = [
            {
                'original_text': ' La Carta de las Naciones Unidas: tratado fundacional de las Naciones Unidas que establece que las obligaciones con las Naciones Unidas prevalecen sobre todas las demás obligaciones del tratado y es vinculante para todos los miembros de las Naciones Unidas.  Tipo de documento: tratado., Fecha de la firma: 26 de junio de 1945., Lugar de la firma: San Francisco, California, Estados Unidos., Entrada en vigor: 24 de octubre de 1945., Firmantes: Ratificado por China, Francia, la Unión Soviética, el Reino Unido, Estados Unidos y por la mayoría de estados signatarios., Artículos: 193., Secciones: 20 (preámbulo y 19 capítulos):, *Preámbulo de la Carta de las Naciones Unidas., *Capítulo I: Propósitos y principios., *Capítulo II: Miembros., *Capítulo III: Órganos, *Capítulo IV: La Asamblea General., *Capítulo V: El Consejo de Seguridad, *Capítulo VI: Solución pacífica de controversias., *Capítulo VII: Acción con respecto a las amenazas a la paz, las rupturas de la paz y los actos de agresión., *Capítulo VIII: Acuerdos Regionales., *Capítulo IX: Cooperación internacional económica y social., *Capítulo X: El Consejo Económico y Social., *Capítulo XI: Declaración sobre los territorios no autónomos., *Capítulo XII: Sistema Internacional de Administración Fiduciaria., *Capítulo XIII: El Consejo de Administración Fiduciaria., *Capítulo XIV: La Corte Internacional de Justicia., *Capítulo XV: La Secretaría., *Capítulo XVI: Disposiciones varias., *Capítulo XVII: Arreglos transitorios de seguridad., *Capítulo XVIII: Enmiendas., *Capítulo XIX: Ratificación y firma. ',
                'gt_entities': [
                    [0, 0, 'Q171328', 'wiki', 4, 28],
                    [0, 0, 'Q131569', 'wiki', 34, 7],
                    [0, 0, 'Q1065', 'wiki', 61, 15],
                    [0, 0, 'Q1065', 'wiki', 120, 15],
                    [0, 0, 'Q131569', 'wiki', 186, 7],
                    [0, 0, 'Q1065', 'wiki', 241, 15],
                    [0, 0, 'Q49848', 'wiki', 267, 9],
                    [0, 0, 'Q384515', 'wiki', 278, 7],
                    [0, 0, 'Q205892', 'wiki', 288, 5],
                    [0, 0, 'Q188675', 'wiki', 300, 5],
                    [0, 0, 'Q2661', 'wiki', 307, 11],
                    [0, 0, 'Q5240', 'wiki', 322, 4],
                    [0, 0, 'Q62', 'wiki', 348, 13],
                    [0, 0, 'Q99', 'wiki', 363, 10],
                    [0, 0, 'Q30', 'wiki', 375, 14],
                    [0, 0, 'Q2955', 'wiki', 410, 13],
                    [0, 0, 'Q148', 'wiki', 460, 5],
                    [0, 0, 'Q7275', 'wiki', 547, 7],
                    [0, 0, 'Q1129448', 'wiki', 601, 9],
                    [0, 0, 'Q1980247', 'wiki', 616, 9],
                    [0, 0, 'Q7239343', 'wiki', 630, 44],
                    [0, 0, 'Q211364', 'wiki', 703, 10],
                    [0, 0, 'Q160016', 'wiki', 730, 8],
                    [0, 0, 'Q895526', 'wiki', 756, 7],
                    [0, 0, 'Q47423', 'wiki', 782, 16],
                    [0, 0, 'Q37470', 'wiki', 817, 20],
                    [0, 0, 'Q1255828', 'wiki', 874, 13],
                    [0, 0, 'Q1728608', 'wiki', 891, 105],
                    [0, 0, 'Q454', 'wiki', 945, 3]
                ],
            },
        ]

        texts = [example['original_text'] for example in examples]
        mention_offsets = [[offset for _, _, _, _, offset,
                            _ in example['gt_entities']] for example in examples]
        mention_lengths = [[length for _, _, _, _, _,
                            length in example['gt_entities']] for example in examples]
        entities = [[0 for _, _, _, _, _, _ in example['gt_entities']]
                    for example in examples]

        batch = {
            "texts": texts,
            "mention_offsets": mention_offsets,
            "mention_lengths": mention_lengths,
            "entities": entities,
        }

        transform = JointELXlmrRawTextTransform()
        model_inputs = transform(batch)

        expected_mention_offsets = [[2,   9,  14,  25,  37,  48,  55,  57,  60,  64,  66,  70,  78,  81, 83,  91, 104, 125, 141, 146, 151, 174, 183, 194, 205, 216, 230]]
        expected_mention_lengths = [[6,  1,  3,  3,  1,  3,  1,  1,  2,  1,  3,  1,  2,  1,  2,  3,  1,  1, 3,  2, 11,  1,  3,  3,  2,  3,  2]]

        self.assertEqual(
            model_inputs['mention_offsets'].tolist(), expected_mention_offsets)
        self.assertEqual(
            model_inputs['mention_lengths'].tolist(), expected_mention_lengths)


if __name__ == '__main__':
    unittest.main()
