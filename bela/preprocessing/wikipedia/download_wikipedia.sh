#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


mkdir wikipedia
cd wikipedia

# new dump
wget http://wikipedia.c3sl.ufpr.br/$enwiki/20210920/$enwiki-20210920-pages-articles-multistream.xml.bz2

# old dump
wget wget http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2

