import pickle
import tqdm
import numpy as np

from collections import defaultdict
from src.util.timer import Timer

"""
This script merges partial results from aoc_part_1.py,
input:
    ├── layer2term2emblist_part_0.pkl
    ├── layer2term2emblist_part_1.pkl
    ├── layer2term2emblist_part_2.pkl
    ├── layer2term2emblist_part_3.pkl
    ├── layer2term2emblist_part_4.pkl
    ├── layer2term2emblist_part_5.pkl
    ├── layer2term2emblist_part_6.pkl
    ├── layer2term2emblist_part_7.pkl
    ├── layer2term2emblist_part_8.pkl
    └── layer2term2emblist_part_9.pkl
output:
term2cs2emb_layer_0.pkl
term2cs2emb_layer_1.pkl
term2cs2emb_layer_2.pkl
......
"""


def aggregate_emblist_by_context_size(emblist):
  emblist = emblist[:61]
  chunksize = 10
  cs2emb = {}
  first_k = 5
  first_five = list(range(1, first_k+1))
  len_emblist = len(emblist)
  context_sizes = first_five + list(range(chunksize, len_emblist + 1, chunksize))
  for context_size in context_sizes:
    if context_size <= len_emblist:
      subset = emblist[:context_size]
      emb = np.mean(subset, axis=0)
      cs2emb[context_size] = emb
    else:
      break

  if len_emblist not in cs2emb:
    cs2emb[len_emblist] = np.mean(emblist, axis=0)

  return cs2emb


def main(model, modelsize, lang, timer):
  print("merging embeddings for %s (model=%s)" % (lang, model))
  layer2term2cs2emb = defaultdict(dict)
  layer2term2emblist = defaultdict(lambda: defaultdict(list))
  casing = "lowercased" if model == "hf_bert-base-multilingual-uncased" else "cased"
  directory = "/work/usr/experiments/EncEval/AOC/%s_wikipedia_clean_full_vocab/" \
              "AOC_cs=60/model=%s/%s/" % (casing, model, lang)

  selected_layers = list(range(modelsize+1)) # +1 for embedding layer 0
  print("processing layer=%s" % selected_layers)
  print("directory: %s" % directory)
  with open(directory + "layerTermEmb_tuples.pkl", "rb") as f:
    try:
      i = 0
      while True:
        (layer, term, emb) = pickle.load(f)

        if i % 1000000 == 0 and i > 0:
          print("loaded %sM tuples (%s)" % (str(int(i/1000000)), timer.pprint_lap()))
        i += 1

        if layer not in selected_layers:
          continue

        if term not in layer2term2cs2emb[layer]:
          layer2term2emblist[layer][term].append(emb)

          if len(layer2term2emblist[layer][term]) == 100:
            cs2emb = aggregate_emblist_by_context_size(layer2term2emblist[layer][term])
            layer2term2cs2emb[layer][term] = cs2emb

    except EOFError:
      print("dataloaders loaded")
  del f
  print("%s records" % str(i))

  for layer, term2emblist in layer2term2emblist.items():
    print("layer=%s" % str(layer))
    term2cs2emb = layer2term2cs2emb[layer]
    for term, emblist in tqdm.tqdm(term2emblist.items(), total=len(term2emblist)):
      if term not in term2cs2emb:
        term2cs2emb[term] = aggregate_emblist_by_context_size(emblist)
    print("aggregation done %s" % timer.pprint_lap())
    with open(directory + "term2cs2emb_layer_%s.pkl" % str(layer), "wb") as f:
      pickle.dump(term2cs2emb, f)
    print("saving done %s" % timer.pprint_lap())
  print("done")


if __name__ == '__main__':
  timer = Timer()
  models = [("hf_xlm-mlm-100-1280", 16),
            ("hf_bert-base-multilingual-uncased", 12)]

  model, modelsize = models[0]
  print("model: %s" % model)
  for lang in ["en", "de", "it", "fi", "ru"]:
    print("\n\nrunning %s\t%s\n\n" % (lang, timer.pprint_lap()))
    main(lang=lang, model=model, modelsize=modelsize, timer=timer)
