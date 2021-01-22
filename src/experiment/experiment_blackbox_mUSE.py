import os
import pickle
import tqdm
import numpy as np
import src.config as c
os.environ["TFHUB_CACHE_DIR"] = c.TF_HUB_DIR
import tensorflow_hub as hub
import tensorflow_text # do not remove this line

from src.experiment.evaluate import layer_wise_evaluation
from src.util.helper import chunk

_mUSE_model = None
def mUSE_blackbox_experiment(query_lang, doc_lang, experiment_data, encode_fn, directory="", batch_size=4, **kwargs):
  global _mUSE_model
  print("running m-USE blackbox experiment")
  if _mUSE_model is None:
    _mUSE_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

  doc_ids, raw_documents, raw_queries, query_ids, relass = experiment_data
  embed = _mUSE_model

  query_representations = []
  chunks = list(chunk(raw_queries, size=10))
  for batch in tqdm.tqdm(chunks, total=len(chunks)):
    current_representations = embed(batch).numpy()
    query_representations.extend(current_representations)
  query_representations = np.expand_dims(query_representations, 0)

  docs_cache = directory + "docs_%s_%s_%s.npy" % (doc_lang, str(len(raw_documents)), str(kwargs["maxlen"]))
  docids_cache = directory + "docids_%s_%s_%s.pkl" % (doc_lang, str(len(raw_documents)), str(kwargs["maxlen"]))
  if not (os.path.exists(docs_cache) and os.path.exists(docids_cache)):
    print("doc embeddings not found, creating new embeddings: %s" % docs_cache)
    tmp_ids = []
    tmp_docs = []
    for _id, doc in zip(doc_ids, raw_documents):
      if doc.strip() != "":
        tmp_ids.append(_id)
        tmp_docs.append(" ".join(doc.split()[:kwargs["maxlen"]]))
    raw_documents = tmp_docs
    doc_ids = tmp_ids

    document_representations = []
    chunks = list(chunk(raw_documents, size=1))
    for batch in tqdm.tqdm(chunks, total=len(chunks)):
      current_representations = embed(batch).numpy()
      document_representations.extend(current_representations)
    document_representations = np.expand_dims(document_representations, 0)

    with open(docs_cache, "wb") as f:
      np.save(f, document_representations)
    with open(docids_cache, "wb") as f:
      pickle.dump(doc_ids, f)
  else:
    print("loading cached doc representations: %s" % docs_cache)
    with open(docs_cache, "rb") as f:
      document_representations = np.load(f)
    with open(docids_cache, "rb") as f:
      doc_ids = pickle.load(f)

  lang_pair = query_lang + "-" + doc_lang
  result = layer_wise_evaluation(doc_ids, document_representations, query_ids, query_representations, relass,
                                 lang_pair=lang_pair,model_dir=directory)
  print("max-len: %s\tmap:%s" % (str(kwargs["maxlen"]), str(result[0])))
  return result
