import logging
import torch

from bela.utils.blink_encoder import BiEncoderRanker
from bela.utils.blink_encoder import encode_candidate, load_or_generate_candidate_pool

logger = logging.getLogger(__name__)


def embed(params):
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

