import os
from src.config import CLWE_BASE_DIR


def get_clwe_directories(selected_embedding_spaces, src, tgt):
  emb_space2folder_path = {"cca": "%s/cca/%s-%s/" % (CLWE_BASE_DIR, src, tgt),
                           "proc": "%s/proc/%s-%s/" % (CLWE_BASE_DIR, src, tgt),
                           "procb": "%s/procb/%s-%s/" % (CLWE_BASE_DIR, src, tgt),
                           "rcsls": "%s/rcsls/%s-%s/" % (CLWE_BASE_DIR, src, tgt),
                           "icp": "%s/icp/%s-%s/" % (CLWE_BASE_DIR, src, tgt),
                           "muse": "%s/muse/%s-%s/" % (CLWE_BASE_DIR, src, tgt),
                           "vecmap": "%s/vecmap/%s-%s/" % (CLWE_BASE_DIR, src, tgt),
                           "mbert_aoc": "%s/mbert_aoc" % CLWE_BASE_DIR,
                           "mbert_iso": "%s/mbert_iso" % CLWE_BASE_DIR,
                           "xlm_aoc": "%s/xlm_aoc" % CLWE_BASE_DIR,
                           "xlm_iso": "%s/xlm_iso" % CLWE_BASE_DIR}
  selected_embedding_spaces = {emb_space: emb_space2folder_path[emb_space] for emb_space in selected_embedding_spaces}
  return selected_embedding_spaces


def get_clwe_file_paths(current_emb_space, folder_path, src, tgt, dataset):
  """
  Optimal layers CLEF:
    mbert_aoc + clef: 9   xlm_aoc + clef/europarl: 15/12
    mbert_iso + clef: 0   xlm_iso + clef: 1
  :param current_emb_space:
  :param folder_path: base directory
  :param src: source language
  :param tgt: target lanuage
  :param dataset: "europarl" or "clef", only relevant for embedding spaces induced by encoders.
  :return: paths to load experiment data
  """
  if current_emb_space == "procb":
    path_query_emb = folder_path + "%s-%s.%s.yacle.train.freq.1k.vectors" % (src, tgt, src)
    path_query_vocab = path_query_emb.replace("vectors", "vocab")
    path_doc_emb = folder_path + "%s-%s.%s.yacle.train.freq.1k.vectors" % (src, tgt, tgt)
    path_doc_vocab = path_doc_emb.replace("vectors", "vocab")
  elif current_emb_space == "rcsls":
    path_query_vocab = folder_path + "fasttext.%s-%s.yacle.train.freq.5k.%s.vec.vocabulary" % (src, tgt, src)
    path_query_emb = path_query_vocab.replace("vocabulary", "embeddings")
    path_doc_vocab = folder_path + "fasttext.%s-%s.yacle.train.freq.5k.%s.vec.vocabulary" % (src, tgt, tgt)
    path_doc_emb = path_doc_vocab.replace("vocabulary", "embeddings")
  elif current_emb_space == "muse" or current_emb_space == "vecmap":
    path_query_emb = folder_path + "fasttext.%s-%s.unsup.%s.vec.embeddings" % (src, tgt, src)
    path_query_vocab = path_query_emb.replace("embeddings", "vocabulary")
    path_doc_emb = folder_path + "fasttext.%s-%s.unsup.%s.vec.embeddings" % (src, tgt, tgt)
    path_doc_vocab = path_doc_emb.replace("embeddings", "vocabulary")
  elif current_emb_space == "icp":
    path_query_emb = folder_path + "proj_src.vectors.npy"
    path_query_vocab = folder_path + "ft.wiki.%s.300.vocab" % src
    path_doc_vocab = folder_path + "ft.wiki.%s.300.vocab" % tgt
    path_doc_emb = folder_path + "ft.wiki.%s.300.vectors" % tgt
  elif current_emb_space == "proc" or current_emb_space == "cca":
    path_query_emb = folder_path + "vectors_%s-%s.%s.yacle.train.freq.5k.np" % (src, tgt, src)
    path_query_vocab = folder_path + "vocab_%s-%s.%s.yacle.train.freq.5k.pkl" % (src, tgt, src)
    path_doc_emb = folder_path + "vectors_%s-%s.%s.yacle.train.freq.5k.np" % (src, tgt, tgt)
    path_doc_vocab = folder_path + "vocab_%s-%s.%s.yacle.train.freq.5k.pkl" % (src, tgt, tgt)
  else:
    assert current_emb_space in ["mbert_aoc", "mbert_iso", "xlm_aoc", "xlm_iso"]
    if current_emb_space.endswith("iso"):
      optimal_layer = 0 if current_emb_space.startswith("mbert") else 1
    else:
      if current_emb_space.startswith("mbert"):
        optimal_layer = 9
      else:
        optimal_layer = 15 if dataset == "clef" else 12
    tgtdir = folder_path + "_layer_%s/%s-%s/" % (str(optimal_layer), src, tgt)
    path_query_emb = tgtdir + "%s-%s.%s.vectors" % (src, tgt, src)
    path_query_vocab = tgtdir + "%s-%s.%s.vocab" % (src, tgt, src)
    path_doc_emb = tgtdir + "%s-%s.%s.vectors" % (src, tgt, tgt)
    path_doc_vocab = tgtdir + "%s-%s.%s.vocab" % (src, tgt, tgt)
  return path_doc_emb, path_doc_vocab, path_query_emb, path_query_vocab

