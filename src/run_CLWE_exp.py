import os
import sys
import argparse
import nltk

from src.dataloaders.paths_clwe import get_clwe_file_paths
from src.dataloaders.load_corpus import prepare_clef_experiment
from src.experiment.experiment_base import run_we_based_experiment
from src.experiment.baselines import run_unigram_lm
from src.experiment.baselines import run_wordbyword_translation
from src.experiment.baselines import run_googletranslate_translation as run_clef_google_translation
from src.dataloaders.load_corpus import load_europarl_corpus
from src.util.embeddings import Embeddings
from src.util.timer import Timer
from src.model.text2vec import text2vec_idf_sum
from src.dataloaders.paths_corpus import get_lang2pair
from src.dataloaders.paths_corpus import all_paths
from src.dataloaders.paths_corpus import PATH_BASE_QUERIES
from src.dataloaders.paths_corpus import PATH_BASE_EVAL
from src.dataloaders.paths_clwe import get_clwe_directories
from src.config import NLTK_DATA


nltk.data.path = [NLTK_DATA]
language_pairs = [("en", "fi"), ("en", "it"), ("en", "de"), ("en", "ru"),
                  ("de", "fi"), ("de", "it"), ("de", "ru"),
                  ("fi", "it"), ("fi", "ru")]
query_limit = None  # set limit for testing/debugging, e.g. 10
doc_limit = None  # set limit for testing/debugging, e.g. 100
year = "2003"
processes = 40

def run(query_lang,  doc_lang, limit=None, path_query_embeddings="", path_query_vocab="", path_doc_embeddings="",
        path_doc_vocab="",  retrieval_method="IDF-SUM", dataset="clef", to_lower=True, emb_space_method="",
        verbose=False):
  """
  Runs for a given language pair, retrieval method and dataset one single CLIR experiment.
  :param query_lang: query language, e.g. "en"
  :param doc_lang: doument language, e.g. "de"
  :param limit: limits queries and documents to the first k instances
  :param path_query_embeddings: path to file for query embeddings (numpy file, shape=[num_embeddings x hidden_size])
  :param path_query_vocab: path to file for query vocabulary (pickle file containing {term: row_id in emb file}
  :param path_doc_embeddings: same as query_path_embeddings
  :param path_doc_vocab: same as query_path_vocab
  :param retrieval_method: options: "IDF-SUM", "TbT-Qt", "gtranslate", "unigram"
  :param dataset: options: "europarl", "clef"
  :param to_lower: whether to lowercase documents and queries (should be always True except for xlm)
  :param emb_space_method: options: "cca", "proc", "procb", "rcsls", "icp", "muse", "vecmap", or encoder variants.
  :param verbose: print progress to stdout.
  :return:
  """
  stdout = sys.stdout
  if not verbose:
    sys.stdout = open(os.devnull, 'w')

  src_lang = get_lang2pair(query_lang)
  tgt_lang = get_lang2pair(doc_lang)
  # Debugging: analysis of out-of-vocabulary terms (1/2)
  # get_oov_size = lambda corpus, lang: [term for term in set(chain(*[doc.split() for doc in corpus])) if
  #                                      term not in embeddings.lang_vocabularies[lang]]
  embeddings = Embeddings()
  if path_query_vocab:
    embeddings.load_serialized_embeddings(path_query_vocab, path_query_embeddings, src_lang[1], limit)
    embeddings.load_serialized_embeddings(path_doc_vocab, path_doc_embeddings, tgt_lang[1], limit)

  if dataset == "clef":
    current_path_queries = PATH_BASE_QUERIES + year + "/Top-" + src_lang[0] + year[-2:] + ".txt"
    current_path_documents = all_paths[tgt_lang[0]][year]
    current_assessment_file = PATH_BASE_EVAL + year + "/qrels_" + tgt_lang[1]
    current_experiment_data = prepare_clef_experiment(current_path_documents, doc_limit, current_path_queries,
                                                      query_limit, query_lang, current_assessment_file)
    # Debugging: analysis of out-of-vocabulary terms (2/2)
    # doc_oov = get_oov_size(corpus=current_experiment_data[1], lang=tgt_lang[1])
    # query_oov = get_oov_size(corpus=current_experiment_data[3], lang=src_lang[1])
    # [term for term in doc_oov if (term[0].upper() + term[1:]) not in embeddings.lang_vocabularies["finnish"]]
    # print("doc OOV: %s\tquery OOV: %s" % (str(len(doc_oov)), str(len(query_oov))))

    if retrieval_method == "TbT-QT":
      evaluation_result = run_wordbyword_translation(query_lang=src_lang, doc_lang=tgt_lang,
                                                     experiment_data=current_experiment_data,
                                                     initialized_embeddings=embeddings)
    elif retrieval_method == "IDF-SUM":
      # with open("/home/user/what_xlm_sees/clef/%s-%s/%s.documents" % (src, tgt, tgt) , "r") as f:
      #   ref_docs = [line.strip() for line in f.readlines()]
      # with open("/home/user/what_xlm_sees/clef/%s-%s/%s.queries" % (src, tgt, src), "r") as f:
      #   ref_queries = [line.strip() for line in f.readlines()]
      # doc_ids, documents, query_ids, queries, relass = current_experiment_data
      # current_experiment_data = (doc_ids, ref_docs, query_ids, ref_queries, relass)
      model_dir = "/".join(path_query_vocab.split("/")[:-1]) + "/"
      evaluation_result = run_we_based_experiment(text2vec_idf_sum,
                                                  query_lang=src_lang,
                                                  doc_lang=tgt_lang,
                                                  experiment_data=current_experiment_data,
                                                  processes=processes,
                                                  initialized_embeddings=embeddings,
                                                  to_lower=to_lower,
                                                  emb_space_method=emb_space_method,
                                                  model_dir=model_dir,
                                                  dataset=dataset)
    elif retrieval_method == "gtranslate":
      evaluation_result = run_clef_google_translation(src_lang, tgt_lang, experiment_data=current_experiment_data)
    elif retrieval_method == "unigram":
      evaluation_result = run_unigram_lm(src_lang, tgt_lang, experiment_data=current_experiment_data)
    else:
      raise NotImplementedError("Method not implemented: " + retrieval_method)
  else:
    assert dataset == "europarl"
    current_experiment_data = load_europarl_corpus(doc_lang=doc_lang, query_lang=query_lang, to_lower=to_lower,
                                                   gtranslate=retrieval_method == "gtranslate")
    if retrieval_method == "IDF-SUM":
      model_dir = "/".join(path_query_vocab.split("/")[:-1]) + "/"
      evaluation_result = run_we_based_experiment(text2vec_idf_sum,
                                                  query_lang=src_lang,
                                                  doc_lang=tgt_lang,
                                                  experiment_data=current_experiment_data,
                                                  processes=processes,
                                                  initialized_embeddings=embeddings,
                                                  to_lower=to_lower,
                                                  emb_space_method=emb_space_method,
                                                  model_dir=model_dir,
                                                  dataset=dataset)
    elif retrieval_method == "TbT-QT":
      evaluation_result = run_wordbyword_translation(query_lang=src_lang, doc_lang=tgt_lang,
                                                     experiment_data=current_experiment_data,
                                                     initialized_embeddings=embeddings)
    elif retrieval_method == "gtranslate" or retrieval_method == "unigram":
      evaluation_result = run_unigram_lm(query_lang=src_lang, doc_lang=tgt_lang, experiment_data=current_experiment_data)
    else:
      raise NotImplementedError("Method not implemented.")

  if not verbose:
    sys.stdout.close()
  sys.stdout = stdout
  return evaluation_result


def run_compact(params):
  return run(**params)


def main():
  if user_args.lang_pairs:
    selected_language_pairs = [(langpair[:2], langpair[2:]) for langpair in user_args.lang_pairs[0]]
  else:
    selected_language_pairs = language_pairs
  if user_args.dataset == "europarl":
    selected_language_pairs = [(l1, l2) for l1, l2 in selected_language_pairs if l1 != "ru" and l2 != "ru"]
  timer = Timer()

  # by default run both retrieval models: [IDF-SUM, TbT-QT]
  selected_retrieval_models = user_args.retrieval_models[0] if user_args.retrieval_models else all_retrieval_models

  # by default run no CLWE space
  selected_embedding_spaces = user_args.emb_spaces[0] if user_args.emb_spaces else None

  # by default run no baseline
  selected_baselines = user_args.baselines[0] if user_args.baselines else None

  langpair2emb_space2map = {}

  print("language pair(s): %s" % selected_language_pairs)
  print("embedding space(s): %s" % selected_embedding_spaces)
  print("dataset: %s" % user_args.dataset)
  print("retrieval method(s): %s" % selected_retrieval_models)

  # run baselines
  if selected_baselines:
    for src, tgt in selected_language_pairs:
      print("running baseline(s): %s->%s" % (src, tgt))
      if user_args.dataset == "europarl" and (src == "ru" or tgt == "ru"):
        continue
      emb_space2map = {}
      for baseline in selected_baselines:
        tag = baseline if not selected_embedding_spaces else baseline + ";-"
        result = run(query_lang=src, doc_lang=tgt, retrieval_method=baseline, dataset=user_args.dataset,
                     to_lower=False, verbose=user_args.verbose)
        print("%s: %s\t%s" % (baseline, str(result), timer.pprint_lap()))
        emb_space2map[tag] = result
      langpair2emb_space2map[src + "-" + tgt] = emb_space2map

    # run embedding space models
  if selected_embedding_spaces:
    for retrieval_method in selected_retrieval_models:
      for src, tgt in selected_language_pairs:
        print("running: %s->%s" % (src, tgt))
        emb_space2map = {}
        embspace_path = get_clwe_directories(selected_embedding_spaces, src, tgt)

        for current_emb_space, folder_path in embspace_path.items():
          to_lower = not current_emb_space.startswith("xlm")
          paths = get_clwe_file_paths(current_emb_space, folder_path, src, tgt, user_args.dataset)
          for path in paths:
            if not os.path.exists(path):
              raise FileNotFoundError("File not found: %s" % path)
          path_doc_emb, path_doc_vocab, path_query_emb, path_query_vocab = paths
          tag = "%s;%s" % (current_emb_space, retrieval_method)
          result = run(query_lang=src,
                       path_query_embeddings=path_query_emb,
                       path_query_vocab=path_query_vocab,
                       doc_lang=tgt,
                       path_doc_embeddings=path_doc_emb,
                       path_doc_vocab=path_doc_vocab,
                       retrieval_method=retrieval_method,
                       dataset=user_args.dataset,
                       emb_space_method=current_emb_space,
                       to_lower=to_lower,
                       verbose=user_args.verbose)
          print("%s: %s\t%s" % (tag, str(result), timer.pprint_lap()))
          emb_space2map[tag] = result
        lp = "%s-%s" % (src, tgt)
        if lp in langpair2emb_space2map:
          langpair2emb_space2map[lp].update(emb_space2map)
        else:
          langpair2emb_space2map[lp] = emb_space2map

  if langpair2emb_space2map:
    # print csv-formatted results
    print(";;", end="")
    for src, tgt in selected_language_pairs:
      print(src + "-" + tgt, end=";")
    print()

    first_lp = list(langpair2emb_space2map.keys())[0]
    evaluated_methods = langpair2emb_space2map[first_lp].keys()
    for emb_space in evaluated_methods:
      print(emb_space, end=";")
      for src, tgt in selected_language_pairs:
        print(str(langpair2emb_space2map[src + "-" + tgt][emb_space]), end=";")
      print()
  print("done with evaluating %s" % user_args.dataset)


if __name__ == "__main__":
  static_emb_spaces = ["cca", "proc", "procb", "rcsls", "icp", "muse", "vecmap"]
  encoder_emb_spaces = ["xlm_aoc",  "mbert_aoc", "xlm_iso", "mbert_iso"]
  emb_spaces = static_emb_spaces + encoder_emb_spaces

  all_lps = [l1 + l2 for l1, l2 in language_pairs]
  all_retrieval_models = ["IDF-SUM", "TbT-QT"]
  all_baselines = ["unigram", "gtranslate"]
  datasets = ["europarl", "clef"]

  parser = argparse.ArgumentParser()
  # mandatory arguments
  parser.add_argument("--dataset", type=str, choices=datasets, required=True)
  # optional arguments
  parser.add_argument("--emb_spaces", action="append", nargs="+", choices=emb_spaces, required=False)
  parser.add_argument("--retrieval_models", type=str, action="append", nargs="+", choices=all_retrieval_models, required=False)
  parser.add_argument("--baselines", action="append", nargs="+", choices=all_baselines, required=False)
  parser.add_argument("--lang_pairs", action='append', nargs="+", choices=all_lps, required=False,
                      help="One or more space-separated language pairs, e.g. 'ende defi'. Defaults to all lang. pairs.")
  parser.add_argument("--verbose", action="store_true", required=False, default=False)
  user_args = parser.parse_args()
  main()
