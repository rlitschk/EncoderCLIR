#!/bin/bash

BASEDIR=$(dirname "$0")
echo "$BASEDIR/data/"

madata="https://madata.bib.uni-mannheim.de/"
for i in "360/9/europarl.tar.gz europarl" \
         "361/3/xlm_iso_layer_1.tar.gz embedding_spaces/xlm_iso_layer_1" \
         "361/4/xlm_aoc_layer_12.tar.gz embedding_spaces/xlm_aoc_layer_12" \
         "361/5/xlm_aoc_layer_15.tar.gz embedding_spaces/xlm_aoc_layer_15" \
         "361/1/mbert_iso_layer_0.tar.gz embedding_spaces/mbert_iso_layer_0" \
         "361/6/checkpoints.tar.gz checkpoints"
do
  set -- $i
  if [ ! -d $BASEDIR/data/$2 ]; then
    wget $madata$1 -P $BASEDIR/data/
    tar -xzvf $BASEDIR/data/$2.tar.gz -C $BASEDIR/data/
    rm $BASEDIR/data/$2.tar.gz
  fi
done
