#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


mkdir wikipedia
cd wikipedia

for LANG in en
do
    wget http://wikipedia.c3sl.ufpr.br/${LANG}wiki/20191001/${LANG}wiki-20191001-pages-articles-multistream.xml.bz2
done

for LANG in en
do
    wikiextractor ${LANG}wikinews-20191001-pages-articles-multistream.xml.bz2 -o ${LANG} --links --lists --sections
done
