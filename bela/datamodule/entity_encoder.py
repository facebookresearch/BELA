import logging
import torch

from bela.utils.blink_encoder import BiEncoderRanker
from bela.utils.blink_encoder import encode_candidate, load_or_generate_candidate_pool

logger = logging.getLogger(__name__)


def embed(path_to_model, entity_dict_path):
    # TO DO: params
    params = {'lower_case': True,
            'path_to_model': path_to_model,
            'data_parallel': False,
            'no_cuda': False,
            'bert_model': 'bert-large-uncased',
            'lowercase': True,
            'out_dim': 1,
            'pull_from_layer': -1,
            'add_linear': False,
            'entity_dict_path': entity_dict_path,
            'debug': False,
            'max_cand_length': 128,
            'encode_batch_size': 32,
            'silent': False,
            }
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    # model = reranker.model

    candidate_pool = load_or_generate_candidate_pool(
        tokenizer,
        params,
    )

    candidate_encoding = encode_candidate(
        reranker,
        candidate_pool,
        params["encode_batch_size"],
        silent=params["silent"],
    )

    output_path = '.'.join(params['entity_dict_path'].split('.')[:-1])
    logger.info("Saved novel entity embeddings to %d", output_path)
    torch.save(candidate_encoding, output_path + ".t7")

    return candidate_encoding

