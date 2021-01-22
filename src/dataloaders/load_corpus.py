import os
from functools import partial

from src.dataloaders.paths_corpus import all_paths
from src.dataloaders.extractors import load_relevance_assessments, load_queries, load_clef_documents
from src.experiment.experiment_base import timer
from src.dataloaders.paths_corpus import lang_pair2europarl_paths
from src.model.text2vec import clean
from src.dataloaders.paths_corpus import get_lang2pair
from src.dataloaders.paths_corpus import PATH_BASE_QUERIES
from src.dataloaders.paths_corpus import PATH_BASE_EVAL


def load_clef_corpus(query_lang, doc_lang, config=None, year = "2003"):
  query_lang = get_lang2pair(query_lang)
  doc_lang = get_lang2pair(doc_lang)

  current_path_queries = PATH_BASE_QUERIES + year + "/Top-" + query_lang[0] + year[-2:] + ".txt"
  current_path_documents = all_paths[doc_lang[0]][year]
  labels = PATH_BASE_EVAL + year + "/qrels_" + doc_lang[1]
  doc_ids, _, documents, queries, query_ids, _, relass = \
    prepare_data(config, labels, current_path_documents, current_path_queries, query_lang, doc_lang)
  return doc_ids, documents, query_ids, queries, relass


def load_europarl_corpus(doc_lang, query_lang, **config):
  # Europarl
  if "ru" == query_lang or "ru" == doc_lang:
    raise BaseException("Russian not available for Europarl")
  if "config" in config:
    config = config["config"]
  query_path, doc_path = lang_pair2europarl_paths[(query_lang, doc_lang)]

  if config.get("gtranslate", False):
    query_path = query_path + ".translated"

  with open(query_path) as f:
    src_lines = [line.strip() for line in f]
  with open(doc_path) as f:
    tgt_lines = [line.strip() for line in f]
  queries, documents = clean_mt_data(src_lines, tgt_lines, to_lower=config["to_lower"],
                                     strip_punctuation=config.get("strip_punctuation", True))
  query_ids = list(map(str, range(len(queries))))
  doc_ids = list(map(str, range(len(documents))))
  relass = {qid: [qid] for qid in query_ids}
  return doc_ids, documents, query_ids, queries, relass


def prepare_clef_experiment(doc_dirs, limit_documents, query_file, limit_queries, query_language, relevance_assessment_file):
  """
  Loads documents, evaluation dataloaders and queries needed to run different experiments on CLEF dataloaders.
  :param doc_dirs: directories containing the corpora for a specific CLEF campaign
  :param limit_documents: for debugging purposes -> limit number of docs loaded
  :param query_file: CLEF Topics (i.e., query) file
  :param limit_queries: for debugging purposes -> limit number of queries loaded
  :param query_language: language of queries
  :param relevance_assessment_file: relevance assesment file
  :return:
  """
  doc_ids, documents = load_documents(doc_dirs, limit_documents)
  print("Documents loaded %s" % (timer.pprint_lap()))
  relass = load_relevance_assessments(relevance_assessment_file)
  print("Evaluation dataloaders loaded %s" % (timer.pprint_lap()))
  query_ids, queries = load_queries(query_file, language_tag=query_language, limit=limit_queries)
  print("Queries loaded %s" % (timer.pprint_lap()))
  return doc_ids, documents, query_ids, queries, relass


def prepare_data(config, path_relevance_assessments, path_documents, path_queries, query_lang, doc_lang):
  if config is None:
    config = {}
  doc_ids, documents, query_ids, queries, relass = \
    prepare_clef_experiment(path_documents, config.get('doc_limit', None),
                            path_queries, config.get('query_limit', None),
                            query_lang[0], path_relevance_assessments)
  query_tag = "_q_%s_%s" % (query_lang, str(len(queries)))
  document_tag = "_d_%s_%s" % (doc_lang, str(len(documents)))
  return doc_ids, document_tag, documents, queries, query_ids, query_tag, relass


def load_documents(doc_dirs, limit_documents):
  """
  Walks through document files and extracts content.
  :param doc_dirs: folder paths containing document files
  :param limit_documents: load only top k documents
  :return:
  """
  if limit_documents is not None:
    limit_documents -= 1
  documents = []
  doc_ids = []
  limit_reached = False
  for doc_dir, extractor in doc_dirs:
    if not limit_reached:
      for file in next(os.walk(doc_dir))[2]:
        if not file.endswith(".dtd"):
          tmp_doc_ids, tmp_documents = load_clef_documents(doc_dir + file, extractor, limit_documents)
          documents.extend(tmp_documents)
          doc_ids.extend(tmp_doc_ids)
        if len(documents) == limit_documents:
          limit_reached = True
          break
  return doc_ids, documents


def clean_mt_data(src, tar, keep_all_tgt=True, **kwargs):
  src_unique = set()
  tar_unique = set()
  src_clean = []
  tar_clean = []

  if "to_lower" in kwargs:
    clean_fn = partial(clean, to_lower=kwargs["to_lower"], strip_punctuation=kwargs.get("strip_punctuation", True))
  else:
    clean_fn = clean

  if keep_all_tgt:
    for i in range(len(src)):
      s = src[i]
      t = tar[i]
      skip_record = False
      if s not in src_unique:
        src_unique.add(s)
      else:
        skip_record = True
      if t not in tar_unique:
        tar_unique.add(t)
      else:
        skip_record = True
      if not skip_record:
        src_clean.append(clean_fn(s))
        tar_clean.append(clean_fn(t))

    for j in range(i+1, len(tar)):
      skip_record = False
      t = tar[j]
      if t not in tar_unique:
        tar_unique.add(t)
      else:
        skip_record = True
      if not skip_record:
        tar_clean.append(clean_fn(t))
  else:
    for s, t in zip(src, tar):
      skip_record = False
      if s not in src_unique:
        src_unique.add(s)
      else:
        skip_record = True
      if t not in tar_unique:
        tar_unique.add(t)
      else:
        skip_record = True
      if not skip_record:
        src_clean.append(clean_fn(s))
        tar_clean.append(clean_fn(t))
  return src_clean, tar_clean