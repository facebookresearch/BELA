#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

set -e
set -u

DATA_DIR="./data"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [[ ! -f mewsli-9.zip ]]; then
    wget http://dl.fbaipublicfiles.com/bela/data/mewsli-9.zip
    unzip mewsli-9.zip
fi

if [[ ! -f mewsli-9-labelled.zip ]]; then
    wget http://dl.fbaipublicfiles.com/bela/data/mewsli-9-labelled.zip
    unzip mewsli-9-labelled.zip
fi

if [[ ! -f mewsli-9-splitted.zip ]]; then
    wget http://dl.fbaipublicfiles.com/bela/data/mewsli-9-splitted.zip
    unzip mewsli-9-splitted.zip
fi

if [[ ! -f aida.zip ]]; then
    wget http://dl.fbaipublicfiles.com/bela/data/aida.zip
    unzip aida.zip
fi

cd ..