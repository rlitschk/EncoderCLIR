import os
import pickle

from datetime import datetime
import argparse
import nltk
import tqdm
import config as c

from functools import partial
from itertools import chain
from itertools import repeat

from src.dataloaders.load_corpus import load_clef_corpus, load_europarl_corpus
from src.model.all_models import load_all_models
from src.util.helper import create_dir

nltk.data.path = [c.NLTK_DATA]

from nltk.tokenize import sent_tokenize
from src.experiment.experiment_base import run_encoder_based_experiment
from src.experiment.experiment_blackbox_DISTIL import distil_blackbox_experiment
from src.experiment.experiment_blackbox_laBSE import laBSE_blackbox_experiment
from src.experiment.experiment_blackbox_LASER import laser_blackbox_experiment
from src.experiment.experiment_blackbox_mUSE import mUSE_blackbox_experiment
from src.dataloaders.paths_corpus import short2pair
from src.dataloaders.paths_corpus import get_lang2pair
from src.model.generic import ModelWrapper
from src.model.aggregation import Modes
from src.model.text2vec import compute_idf_weights
from src.model.text2vec import clean
from src.util.retrieval_methods import RetrievalMethod
from src.config import DISTIL_TAGS

scale_idf = True
wp_aggr = Modes.FIRST
word_aggr = Modes.SUM

en_x = [("en", "fi"), ("en", "it"), ("en", "ru"), ("en", "de")] #
fi_x = [("fi", "it"), ("fi", "ru")]
de_x = [("de", "fi"), ("de", "it"), ("de", "ru")]
language_pairs = [lang_pair for elem in [en_x, de_x, fi_x] for lang_pair in elem]


def encode(lines, keepwords, tokenizer, config, vocab2idf=None, pad_to_max_length=True):
  """
  Transforms either list of queries/documents into their word-pieces and constructs corresponding idf-value sequences.
  :param lines: queries or documents
  :param keepwords: @deprecated
  :param tokenizer: model tokenizer
  :param config: must contain attributes int maxlen and bool add_special_token
  :param vocab2idf: {term: idf}
  :param pad_to_max_length: if set pads all sequences to maxlen
  :return:
  """
  tmp_wordpiece_ids, tmp_lengths, tmp_masks, tmp_idfs, tmp_words = [], [], [], [], []
  for i, line in tqdm.tqdm(enumerate(lines), total=len(lines)):
    seq, seq_len, mask, idfs, words = tokenizer.encode(line, max_length=config['maxlen'],
                                                       add_special_token=config["add_special_token"],
                                                       keepwords=keepwords, vocab2idf=vocab2idf,
                                                       pad_to_max_length=pad_to_max_length)
    tmp_words.append(words)
    if idfs:
      tmp_idfs.append(idfs)
    if mask:
      tmp_wordpiece_ids.append(seq)
      tmp_lengths.append(seq_len)
      tmp_masks.append(mask)

  if not tmp_idfs:
    tmp_idfs = repeat([])

  return tmp_wordpiece_ids, tmp_lengths, tmp_masks, tmp_idfs, tmp_words


def run_encoder_clir(query_lang, doc_lang, encoder, wp_aggr, word_aggr,
                     model_dir, data_dir, config, tokenizer, dataset,
                     lang2keepwords=None):
  """
  Load data, prepare experiments, run experiments.
  :param query_lang: e.g. EN
  :param doc_lang: e.g. DE
  :param encoder: either ModelWrapper instance or model_str
  :param wp_aggr: aggregation function to aggregate word-pieces into words, e.g. Modes.FIRST
  :param word_aggr: aggregation function to aggregate words into sentences embeddings, e.g. Modes.MEAN
  :param model_dir: target directory to save results
  :param data_dir: source directory to load corpus from
  :param config: additional model parameters (see argparse arguments)
  :param tokenizer: model tokenizer/word-piece-encoder
  :param dataset: either "clef" or "europarl"
  :param lang2keepwords: @deprecated
  :return: {layer: (Mean-Avg-Precision, p-value)}
  """
  print("\nrunning %s->%s" % (query_lang, doc_lang))
  print("%s\t\t(model directory)" % model_dir)
  print("%s\t\t(data directory)" % data_dir)
  config['wp_aggr'] = wp_aggr
  config['word_aggr'] = word_aggr
  # exception case: tokenizer for sbert always adds special token, no need to manually add it (it'd lead to unexpected results)
  config['add_special_token'] = False if type(encoder) == str else config['add_special_token']
  config['pad_to_max_length'] = False if type(encoder) == str else config['add_special_token']
  if config['split_documents']:
    config['strip_punctuation'] = False

  # Load datasets
  load_corpus = load_clef_corpus if dataset == "clef" else load_europarl_corpus
  doc_ids, documents, query_ids, queries, relass = load_corpus(query_lang=query_lang, doc_lang=doc_lang, config=config)

  if config['split_documents']:
    pseudo_docs = []
    pseudo_docids = []
    lang_ = get_lang2pair(doc_lang)[1]
    print("splitting documents into sentence")
    for doc_id, document in tqdm.tqdm(zip(doc_ids, documents), total=len(documents)):
      for i, sentence in enumerate(sent_tokenize(document, language=lang_)):
        pseudo_docs.append(sentence)
        pseudo_docids.append("%s_%s" % (str(doc_id), str(i)))
    documents = pseudo_docs
    doc_ids = pseudo_docids

  if encoder == "muse":
    return mUSE_blackbox_experiment(query_lang, doc_lang, (doc_ids, documents, queries, query_ids, relass),
                                    encode_fn=None, directory=model_dir, maxlen=config['maxlen'])

  if encoder == "labse":
    return laBSE_blackbox_experiment(query_lang=query_lang, doc_lang=doc_lang,
                                     experiment_data=(doc_ids, documents, queries, query_ids, relass),
                                     encode_fn=None, directory=model_dir, **config)

  if encoder == "laser":
    raw_documents = [clean(d, to_lower=config['to_lower']) for d in documents]
    raw_queries = [clean(q, to_lower=config['to_lower']) for q in queries]
    experiment_data = doc_ids, raw_documents, query_ids, raw_queries, relass
    return laser_blackbox_experiment(query_lang=query_lang, doc_lang=doc_lang, experiment_data=experiment_data,
                                     encode_fn=None, directory=model_dir, **config)

  if encoder in DISTIL_TAGS:
    raw_documents = [clean(d, to_lower=config['to_lower']) for d in documents]
    raw_queries = [clean(q, to_lower=config['to_lower']) for q in queries]
    experiment_data = doc_ids, raw_documents, query_ids, raw_queries, relass
    return distil_blackbox_experiment(query_lang=query_lang, doc_lang=doc_lang, experiment_data=experiment_data,
                                     encode_fn=encoder, directory=model_dir, **config)

  # Clean and encode queries
  print("cleaning queries")
  raw_queries = [clean(q, to_lower=config["to_lower"], strip_punctuation=config.get("strip_punctuation", True)) for q in queries]
  print("encoding queries")
  encoded_queries = encode(queries,
                           keepwords=None,
                           tokenizer=tokenizer, config=config,
                           pad_to_max_length=config.get("pad_to_max_length", True))
  queries = *encoded_queries, raw_queries

  # Clean and encode corpus, compute IDF values
  docId_docs = list(zip(range(len(documents)), documents))
  unique_terms = set(chain(*[clean(doc, to_lower=True).split() for doc in documents]))
  corpus_vocab = { short2pair[doc_lang][1]: {unique_term: i for i, unique_term in enumerate(unique_terms)}}
  corpus_path = data_dir + doc_lang + "_" + str(config["maxlen"]) + ".pkl"
  if os.path.exists(corpus_path):
    print("loading corpus")
    with open(corpus_path, "rb") as f:
      documents = pickle.load(f)
  else:
    print("cleaning corpus")
    documents = [clean(d,
                       to_lower=config["to_lower"],
                       strip_punctuation=config.get("strip_punctuation", True)) for d in documents]
    idf_weights = None
    if scale_idf:
      idf_weights = compute_idf_weights(docId_docs, language=get_lang2pair(doc_lang)[1], processes=config["processes"],
                                        embedding_lookup=corpus_vocab, to_lower=config["to_lower"])
    print("encoding corpus")
    corpus = encode(lines=documents,
                    keepwords=None,
                    tokenizer=tokenizer,
                    config=config,
                    vocab2idf=idf_weights,
                    pad_to_max_length=config.get("pad_to_max_length", True))
    corpus = *corpus, documents
    os.makedirs(data_dir, exist_ok=True)
    with open(corpus_path, "wb") as f:
      pickle.dump(corpus, f)
    documents = corpus

  # run experiment
  encode_fn = partial(encoder, wp_aggr=wp_aggr, word_aggr=word_aggr) if type(encoder) != str else encoder
  assert config["to_lower"] != tokenizer.is_cased
  experiment_data = doc_ids, documents, query_ids, queries, relass
  evaluation_result = run_encoder_based_experiment(query_lang=query_lang,
                                                   doc_lang=doc_lang,
                                                   experiment_data=experiment_data,
                                                   encode_fn=encode_fn,
                                                   retrieval_method=RetrievalMethod.COSINE,
                                                   directory=model_dir,
                                                   **config)
  return evaluation_result


def run_single_model(model_name, Model, Tokenizer, config):
  config['name'] += "/" if not config['name'].endswith("/") else ""
  base_dir = c.OUTPUT_DIR + config['name'] + "model=%s/dataset=%s/" % (model_name, str(config['dataset']))
  os.makedirs(base_dir, exist_ok=True)

  print("loading model and tokenizer")
  if type(Model) != str:
    # SEMB models
    tokenizer = Tokenizer()
    encoder = ModelWrapper(Model())
    config['to_lower'] = False if tokenizer.is_cased else True
  else:
    # similarity specialized models
    encoder = Model # model loaded from model str
    tokenizer = None # blackbox models have built in tokenizer
    config['to_lower'] = True if Model in ['laser', 'labse'] else False

  selected_lang_pairs = [(langpair[:2], langpair[2:]) for langpair in config['lang_pairs'][0]] if config['lang_pairs'] else language_pairs
  if config["dataset"] == "europarl":
    selected_lang_pairs = [(l1,l2) for l1, l2 in selected_lang_pairs if l1 != "ru" and l2 != "ru"]
  print("language pairs: %s" % selected_lang_pairs)
  results = {}
  for src, tgt in selected_lang_pairs:
    if config['dataset'] == 'europarl' and (src == "ru" or tgt == "ru"):
      continue

    model_directory = base_dir + "sent=%s/word=%s/" % (
      word_aggr.name, wp_aggr.name)
    data_directory = base_dir + "add_special_token=%s/to_lower=%s/len=%s/rm_stopwords=%s/rm_digits=%s/" % (
      str(config['add_special_token']),
      str(config['to_lower']),
      str(config['maxlen']),
      str(config['rm_stopwords']),
      str(config['rm_digits']))

    lang_pair_str = src + "-" + tgt
    if config['dataset'] == 'europarl':
      model_directory += "%s/" % lang_pair_str
      data_directory += "%s/" % lang_pair_str
    create_dir(model_directory)
    create_dir(data_directory)

    result = run_encoder_clir(src, tgt,
                              encoder=encoder,
                              model_dir=model_directory,
                              data_dir=data_directory,
                              wp_aggr=wp_aggr,
                              word_aggr=word_aggr,
                              config=config,
                              tokenizer=tokenizer,
                              dataset=config['dataset'])
    results[src+tgt] = result

  num_models = 12
  num_langpairs = 9 if config['dataset'] == 'clef' else 6
  bonferroni = num_models * num_langpairs
  if results:
    print_store_results(base_dir, model_name, results, write_file=False, bonferroni=bonferroni)
  else:
    print("Warning: no experiments were run.")
  return base_dir, results


def print_store_results(base_dir, name, layer2map_results, write_file=False, bonferroni=1, alpha=0.05, SEP="\t"):
  """
  Formats results into TSV, prints it (adding significance markers), and optionally writes it to base_dir.

  :param base_dir: File where to write results.
  :param name: arbitrary name for your experiment.
  :param layer2map_results: e.g. {3: {0.31, 0.005}} refers to model layer 3, mean avg precision 0.31 at p-value 0.005.
  :param write_file: whether to serialize onto disk, otherwise print only.
  :param bonferroni: if > 1 then it corrects significance values.
  :param alpha: sinificance level
  :param SEP: csv separator
  :return:
  """
  summary_lines = ["Model=%s\n" % name]
  for langpair, layer2result in layer2map_results.items():
    record = langpair + SEP
    print(record, end="")
    summary_lines.append(record)
  print()

  num_layers = len(layer2map_results[list(layer2map_results.keys())[0]])
  for layer in range(num_layers):
    langpair_directory = []
    for _, layer2result in layer2map_results.items():
      if type(layer2result[layer]) != tuple:
        langpair_directory.append(layer2result[layer])
      else:
        MAP, pvalue = layer2result[layer]
        alpha = alpha / bonferroni
        if pvalue <= alpha:
          MAP = str(round(MAP, 4)) + "*"
        langpair_directory.append(MAP)

    summary_line = SEP.join([str(round(MAP, 4)) if type(MAP) != str else MAP for MAP in langpair_directory])
    print(summary_line)
    summary_lines.append(summary_line)

  if write_file:
    with open(base_dir + "results_%s.txt" % datetime.now().strftime("%d.%m.%Y_%H:%M:%S"), "a") as f:
      f.writelines([line + "\n" for line in summary_lines])


def main(config):
  MODELS = load_all_models()
  all_model_results = {}
  for model in [MODELS[config['encoder']]]:
    model_name, Model, Tokenizer = model
    model_dir, results = run_single_model(model_name, Model, Tokenizer, config)
    all_model_results[model_name] = (results, model_dir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # global parameters (required)
  parser.add_argument("--processes", type=int, required=True, help="Number of parallel processes for IDF calculation.")
  parser.add_argument("--gpu", type=str, required=True, help="Value for CUDA_VISIBLE_DEVICES environment variable.")
  parser.add_argument("--name", type=str, required=True, help="Any arbitrary experiment name.")
  parser.add_argument("--dataset", type=str, required=True, choices=['europarl', 'clef'])
  parser.add_argument("--encoder", type=str, required=True, help="Any SEMB or similarity-specialized encoder.",
                      choices=['mbert', 'xlm', 'laser', 'labse', 'muse', 'distil_mbert', 'distil_xlmr', 'distil_muse'])
  all_lps = [l1 + l2 for l1, l2 in language_pairs]
  parser.add_argument("--lang_pairs", action='append', nargs="+", choices=all_lps,
                      help="One or more space-separated language pairs, e.g. 'ende defi'. Defaults to all lang. pairs.")
  parser.add_argument("--maxlen", type=int, default=128)
  # parser.add_argument("--split_documents", action='store_true', help="Turn on individual sentence scoring.")

  # model parameters
  user_args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = user_args.gpu
  user_args = vars(user_args)
  user_args["batch_size"] = 10
  user_args["add_special_token"] = True
  user_args["rm_punctuation"] = False
  user_args["rm_stopwords"] = False
  user_args["rm_digits"] = False
  main(user_args)
