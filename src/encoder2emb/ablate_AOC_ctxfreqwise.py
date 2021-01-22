from src.run_ENC_exp import print_store_results
import os
import argparse
import pickle
import numpy as np

# install https://github.com/codogogo/xling-eval
from projection import project_proc

from collections import defaultdict
from src.util.timer import Timer
from src.run_CLWE_exp import run as run_static_xling
from src.dataloaders.paths_bli import lang_pair2bli_dict

PROC_B_LIMIT = 5000
MAPPING_TYPE = "proc"
DATASET = "clef"

model_layers = [("hf_xlm-mlm-100-1280", 17),
                ("hf_bert-base-multilingual-cased", 13),
                ("hf_bert-base-multilingual-uncased", 13),  # 2
                ("hf_xlm-mlm-100-1280", 17),  # 3
                ("hf_xlm-roberta-base", 13),
                ("hf_xlm-roberta-large", 25)]

mbert=2
xlm=3
modelname, max_layers = model_layers[mbert]



def get_cs_or_max(cs2emb, context_size):
  """

  :param cs2emb:
  :param context_size:
  :return:
  """
  if context_size in cs2emb:
    return cs2emb[context_size]
  else:
    _max = max(cs2emb.keys())
    return cs2emb[_max]


def load_embs(language, selected_layer, context_size):
  src_dir = base_dir + "%s/" % language
  src_file = src_dir + "term2cs2emb_layer_%s.pkl" % str(selected_layer)
  with open(src_file, "rb") as f:
    term2cs2emb = pickle.load(f)
  term2emb = {k: get_cs_or_max(cs2emb, context_size=context_size) for k, cs2emb in term2cs2emb.items()}
  term2cs = {}
  for term, cs2emb in term2cs2emb.items():
    cs = context_size if context_size in cs2emb else max(cs2emb.keys())
    term2cs[term] = cs
  return term2emb, term2cs


def run_single_lp_cs_compact(args):
  context_size, current_layer, src, tgt, to_lower = args
  return run_single_lp_cs(context_size, current_layer, src, tgt, to_lower)


def save(path, obj):
  with open(path, "wb") as f:
    pickle.dump(obj, f)


def run_single_lp_cs(context_freq, current_layer, lang_src, lang_trg, to_lower):
  """
  Runs static CLIR on a single language pair for a single layer. Evaluates a specific context frequency
  :param context_freq: Max number of contexts selected for each vocabulary entry
  :param current_layer: selected layer of mBERT or XLM
  :param lang_src: source language
  :param lang_trg: target language
  :param to_lower: whether to lowercase queries and documents
  :return:
  """

  tgt_dir = proc_dir + "cs=%s/%s-%s/layer=%s/" % (str(context_freq), lang_src, lang_trg, str(current_layer))
  print(tgt_dir)

  os.makedirs(tgt_dir, exist_ok=True)
  path_src_vocab = tgt_dir + lang_src + "-" + lang_trg + "." + lang_src + ".vocab"
  src_path_embs = tgt_dir + lang_src + "-" + lang_trg + "." + lang_src + ".vectors"
  path_trg_vocab = tgt_dir + lang_src + "-" + lang_trg + "." + lang_trg + ".vocab"
  path_trg_embs = tgt_dir + lang_src + "-" + lang_trg + "." + lang_trg + ".vectors"
  path_proj = tgt_dir + lang_src + "-" + lang_trg + ".proj"
  path_src_term2cs = tgt_dir + lang_src + "-" + lang_trg + "." + lang_src + "_term2cs.pkl"
  path_trg_term2cs = tgt_dir + lang_src + "-" + lang_trg + "." + lang_trg + "_term2cs.pkl"
  paths = [path_src_vocab, src_path_embs, path_trg_vocab, path_trg_embs]

  all_files_exist = True
  for tmp_path in paths:
    if not os.path.exists(tmp_path):
      all_files_exist = False
      print("file not found: %s" % tmp_path)
      break

  if not all_files_exist:
    src_term2emb, src_term2cs = load_embs(lang_src, current_layer, context_freq)
    print("src embs loaded")
    tgt_term2emb, tgt_term2cs = load_embs(lang_trg, current_layer, context_freq)
    print("tgt embs loaded")

    proj_mat = None
    if MAPPING_TYPE == "proc":
      proj_mat, src_emb, src_vocab, tgt_emb, tgt_vocab = map_proc(lang_src, lang_trg, src_term2emb, tgt_term2emb, to_lower)
    else:
      raise NotImplementedError
      # assert MAPPING_TYPE == "procB"
      # _, src_emb, src_vocab, tgt_emb, tgt_vocab = map_procB(lang_src, lang_trg, src_term2emb, tgt_term2emb)

    print("src term2cs path: %s" % path_src_term2cs)
    save(path_src_term2cs, src_term2cs)
    print("trg term2cs path: %s" % path_trg_term2cs)
    save(path_trg_term2cs, tgt_term2cs)

    print("src vocab path: %s" % path_src_vocab)
    save(path_src_vocab, src_vocab)
    print("src emb path: %s" % src_path_embs)
    with open(src_path_embs, "wb") as f:
      np.save(f, src_emb)

    print("trg vocab path: %s" % path_trg_vocab)
    save(path_trg_vocab, tgt_vocab)
    print("tar emb path: %s" % path_trg_embs)
    with open(path_trg_embs, "wb") as f:
      np.save(f, tgt_emb)
    if proj_mat is not None:
      with open(path_proj, "wb") as f:
        np.save(f, proj_mat)

  else:
    print("files exist already, skipping procrustes")

  map_score = run_static_xling(query_lang=lang_src,
                               path_query_vocab=path_src_vocab, path_query_embeddings=src_path_embs,
                               doc_lang=lang_trg,
                               path_doc_vocab=path_trg_vocab, path_doc_embeddings=path_trg_embs,
                               dataset=DATASET,
                               to_lower=to_lower)
  return lang_src, lang_trg, context_freq, current_layer, map_score


def map_proc(lang_src, lang_trg, src_term2emb, tgt_term2emb, do_lowercasing):
  """
  Projects the source embedding space to the target space using procrustes.
  :param lang_src: source language, e.g. EN
  :param lang_trg: target language, e.g. DE
  :param src_term2emb: embedding table for source language
  :param tgt_term2emb: embedding table for target language
  :param do_lowercasing: whether to lowercase embeddings
  :return:
  """
  proj_mat = _get_proc_projectionmatrix(lang_src, lang_trg, src_term2emb, tgt_term2emb, do_lowercasing)
  src_vocab = {}
  src_emb = []
  for i, (term, emb) in enumerate(src_term2emb.items()):
    src_vocab[term] = i
    src_emb.append(emb)

  # apply procrustes on full vocabulary
  src_emb = np.array(src_emb, dtype=np.float32)
  src_emb = np.matmul(src_emb, proj_mat)

  tgt_vocab = {}
  tgt_emb = []
  for i, (term, emb) in enumerate(tgt_term2emb.items()):
    tgt_vocab[term] = i
    tgt_emb.append(emb)

  # already "mapped", i.e. src is mapped onto target space
  tgt_emb = np.array(tgt_emb, dtype=np.float32)
  return proj_mat, src_emb, src_vocab, tgt_emb, tgt_vocab


def _get_proc_projectionmatrix(lang_src, lang_trg, src_term2emb, tgt_term2emb, do_lowercasing):
  """
  Loads the BLI dataset, performs lowercasing if needed, and computes the orthogonal mapping matrix.
  :param lang_src: source language, e.g. EN
  :param lang_trg: target lanugage, e.g. DE
  :param src_term2emb: source embedding table
  :param tgt_term2emb: target embedding table
  :param do_lowercasing: whether to lowercase
  :return:
  """
  bli_train_path = lang_pair2bli_dict[(lang_src, lang_trg)]
  with open(bli_train_path) as f:
    bli_dict_train = [line.strip().split("\t") for line in f.readlines()]

  if do_lowercasing:
    uniques = set()
    translation_pairs = []
    for t1, t2 in bli_dict_train:
      t1 = t1.lower()
      t2 = t2.lower()
      key = t1+t2
      if key not in uniques:
        translation_pairs.append((t1,t2))
        uniques.add(key)
    bli_dict_train = translation_pairs

  vocab_dict_src = {}
  src_embs = []
  for i, (term, emb) in enumerate(src_term2emb.items()):
    vocab_dict_src[term] = i
    src_embs.append(emb)

  vocab_dict_tgt = {}
  tgt_embs = []
  for i, (term, emb) in enumerate(tgt_term2emb.items()):
    vocab_dict_tgt[term] = i
    tgt_embs.append(emb)
  embs_src_projected, proj_mat, _ = project_proc(vocab_dict_src, src_embs, vocab_dict_tgt, tgt_embs, bli_dict_train)
  return proj_mat


def process_langpairs_layers(layer_start, layer_end, context_freq, to_lower):
  """
  Run all language pairs: (1) Map embedding space and (2) Run CLIR
  :param layer_start: Must be in range [0,17] for XLM and [0,13] for mBERT
  :param layer_end: Until which layer to evaluate
  :param context_freq: How many contextualized embeddings should be considered for AOC embeddings
  :param to_lower: whether to lowercase
  :return:
  """
  configuratios = []
  layers = list(range(layer_start, layer_end))
  # layers.reverse()
  for current_layer in layers:
    for src, tgt in lang_pairs:
      if DATASET == "europarl" and (src == "ru" or tgt == "ru"):
        continue
      configuratios.append((context_freq, current_layer, src, tgt, to_lower))

  layer2langpair2map = defaultdict(dict)
  total_configs = len(configuratios)
  for i, config in enumerate(configuratios):
    src, tgt, context_freq, current_layer, map_score = run_single_lp_cs_compact(config)
    layer2langpair2map[current_layer][(src, tgt)] = map_score
    print("\n%s out of %s configurations done!\t(%s)\n" % (str(i+1), str(total_configs), timer.pprint_lap()))

  return layer2langpair2map


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-cs", "--context_size", type=int, required=True)

  args = parser.parse_args()


  timer = Timer()

  casing = "cased" if modelname != "hf_bert-base-multilingual-uncased" else "lowercased"
  to_lower = True if modelname == "hf_bert-base-multilingual-uncased" else False
  print("evaluate %s on %s" % (modelname, DATASET))
  home = "/path/to/embeddings/before/procrustes/"

  context_sizes = [60]
  if modelname == "hf_bert-base-multilingual-uncased":
    base_dir = home + "AOC/%s_wikipedia_clean_full_vocab/AOC_cs=60/model=%s/" % (casing, modelname)
    layer_start = 9
  else:
    base_dir = home + "AOC/%s_wikipedia_clean_full_vocab/AOC_cs=60/model=%s/" % (casing, modelname)
    layer_start = 15 if DATASET == "clef" else 12

  layer_end = layer_start + 1
  lang_pairs = [("en", "fi"), ("en", "it"), ("en", "ru"), ("en", "de"),
                ("de", "fi"), ("de", "it"), ("de", "ru"),
                ("fi", "it"), ("fi", "ru")]

  proc_dir = base_dir + "mapped_%s/" % MAPPING_TYPE

  # layer_start = 0
  # layer_end = max_layers
  # context_sizes = list(range(1, 6)) + list(range(10, 61, 10))
  # context_sizes.reverse()

  user_specified_context_size = args.context_size
  assert user_specified_context_size in context_sizes
  layer2langpair2map = process_langpairs_layers(layer_start, layer_end, context_freq=user_specified_context_size, to_lower=to_lower)

  num_models = 12
  num_langpairs = 9 if DATASET == "clef" else 6
  bonferroni = num_models * num_langpairs
  for layer, langpair2map in layer2langpair2map.items():
    print_store_results(base_dir=proc_dir, name=modelname, layer2map_results=langpair2map, bonferroni=bonferroni)
