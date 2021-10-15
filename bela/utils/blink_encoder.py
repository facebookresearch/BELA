import torch
import torch.nn as nn
from tqdm import tqdm
import json

from pytorch_transformers.modeling_bert import BertModel
from pytorch_transformers.tokenization_bert import BertTokenizer

from torch.utils.data import DataLoader, SequentialSampler


ENT_START_TAG = "[unused0]"
ENT_END_TAG = "[unused1]"
ENT_TITLE_TAG = "[unused2]"
NULL_IDX = 0


class BertEncoder(nn.Module):
    def __init__(
            self, bert_model, output_dim, layer_pulled=-1, add_linear=None):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask):
        output_bert, output_pooler = self.bert_model(
            token_ids, segment_ids, attention_mask
        )
        # get embedding of [CLS] token
        if self.additional_linear is not None:
            embeddings = output_pooler
        else:
            embeddings = output_bert[:, 0, :]

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params['bert_model'])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

    def forward(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
    ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict)

    def build_model(self):
        self.model = BiEncoderModule(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        if random_negs:
            # train on random negatives
            return embedding_ctxt.mm(embedding_cands.t())
        else:
            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, context_input, cand_input, label_input=None):
        flag = label_input is None
        scores = self.score_candidate(context_input, cand_input, flag)
        bs = scores.size(0)
        if label_input is None:
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            loss = F.cross_entropy(scores, target, reduction="mean")
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # TODO: add parameters?
            loss = loss_fct(scores, label_input)
        return loss, scores



def load_entity_dict(params):

    path = params.get("entity_dict_path", None)
    assert path is not None, "Error! entity_dict_path is empty."

    entity_list = []
    #logger.info("Loading entity description from path: " + path)
    with open(path, 'rt') as f:
        for line in f:
            sample = json.loads(line.rstrip())
            title = sample['title']
            text = sample.get("text", "").strip()
            entity_list.append((title, text))
            if params["debug"] and len(entity_list) > 200:
                break

    return entity_list


def get_candidate_representation(
    candidate_desc,
    tokenizer,
    max_seq_length,
    candidate_title=None,
    title_tag=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)
        cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


def get_candidate_pool_tensor(
    entity_desc_list,
    tokenizer,
    max_seq_length,
):
    # TODO: add multiple thread process
    #logger.info("Convert candidate text to id")
    cand_pool = []
    for entity_desc in tqdm(entity_desc_list):
        if type(entity_desc) is tuple:
            title, entity_text = entity_desc
        else:
            title = None
            entity_text = entity_desc

        rep = get_candidate_representation(
                entity_text,
                tokenizer,
                max_seq_length,
                title,
        )
        cand_pool.append(rep["ids"])

    cand_pool = torch.LongTensor(cand_pool)
    return cand_pool


def get_candidate_pool_tensor_helper(
    entity_desc_list,
    tokenizer,
    max_seq_length,
):
    return get_candidate_pool_tensor(
        entity_desc_list,
        tokenizer,
        max_seq_length,
    )


def load_or_generate_candidate_pool(
    tokenizer,
    params,
    cand_pool_path=None,
):
    candidate_pool = None
    if cand_pool_path is not None:
        # try to load candidate pool from file
        try:
            #logger.info("Loading pre-generated candidate pool from: ")
            #logger.info(cand_pool_path)
            candidate_pool = torch.load(cand_pool_path)
        except:
            #logger.info("Loading failed. Generating candidate pool")
            pass

    if candidate_pool is None:
        # compute candidate pool from entity list
        entity_desc_list = load_entity_dict(params)
        candidate_pool = get_candidate_pool_tensor_helper(
            entity_desc_list,
            tokenizer,
            params["max_cand_length"],
        )

        if cand_pool_path is not None:
            #logger.info("Saving candidate pool.")
            torch.save(candidate_pool, cand_pool_path)

    return candidate_pool


def get_candidate_representation(
        candidate_desc,
        tokenizer,
        max_seq_length,
        candidate_title=None,
        title_tag=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)
        cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask


def encode_candidate(
        reranker,
        candidate_pool,
        encode_batch_size,
        silent,
):

    reranker.model.eval()
    device = reranker.device
    sampler = SequentialSampler(candidate_pool)
    data_loader = DataLoader(
        candidate_pool, sampler=sampler, batch_size=encode_batch_size
    )
    if silent:
        iter_ = data_loader
    else:
        iter_ = tqdm(data_loader)

    cand_encode_list = None
    for step, batch in enumerate(iter_):
        cands = batch
        cands = cands.to(device)
        cand_encode = reranker.encode_candidate(cands)
        if cand_encode_list is None:
            cand_encode_list = cand_encode
        else:
            cand_encode_list = torch.cat((cand_encode_list, cand_encode))

    return cand_encode_list
