#!/bin/bash

BASEDIR=$(dirname "$0")
echo "$BASEDIR/data/"

madata="https://madata.bib.uni-mannheim.de/"

for i in "360/9/europarl.tar.gz europarl" \
         "360/3/cca.tar.gz embedding_spaces/cca" \
         "360/2/proc.tar.gz embedding_spaces/proc" \
         "360/7/procb.tar.gz embedding_spaces/procb" \
         "360/6/rcsls.tar.gz embedding_spaces/rcsls" \
         "360/8/vecmap.tar.gz embedding_spaces/vecmap" \
         "360/1/muse.tar.gz embedding_spaces/muse" \
         "360/4/icp.tar.gz embedding_spaces/icp"
do
  set -- $i
  if [ ! -d $BASEDIR/data/$2 ]; then
    wget $madata$1 -P $BASEDIR/data/
    tar -xzvf $BASEDIR/data/$2.tar.gz -C $BASEDIR/data/
    rm $BASEDIR/data/$2.tar.gz
  fi
done