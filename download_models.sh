#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

set -e
set -u

MODELS_DIR="./models"

mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

if [[ ! -f model_aida.ckpt ]]; then
    wget http://dl.fbaipublicfiles.com/bela/models/model_aida.ckpt
fi

if [[ ! -f model_mewsli.ckpt ]]; then
    wget http://dl.fbaipublicfiles.com/bela/models/model_mewsli.ckpt
fi

if [[ ! -f model_wiki.ckpt ]]; then
    wget http://dl.fbaipublicfiles.com/bela/models/model_wiki.ckpt
fi

if [[ ! -f model_e2e.ckpt ]]; then
    wget http://dl.fbaipublicfiles.com/bela/models/model_e2e.ckpt
fi

if [[ ! -f index.txt ]]; then
    wget http://dl.fbaipublicfiles.com/bela/embeddings/index.txt
fi

if [[ ! -f embeddings.pt ]]; then
    wget http://dl.fbaipublicfiles.com/bela/embeddings/embeddings.pt
fi

cd ..