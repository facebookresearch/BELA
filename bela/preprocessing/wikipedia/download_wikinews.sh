#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


mkdir wikinews
cd wikinews

for LANG in en
do
    curl http://wikipedia.c3sl.ufpr.br/${LANG}wikinews/20210901/${LANG}wikinews-20210901-pages-articles-multistream.xml.bz2 --output enwikinews-20210901-pages-articles-multistream.xml.bz2ˇ
done

for LANG in en
do
    wikiextractor ${LANG}wikinews-20210901-pages-articles-multistream.xml.bz2 -o ${LANG} --links --lists --sections
done
