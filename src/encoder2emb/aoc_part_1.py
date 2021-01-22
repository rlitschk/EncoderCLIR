import os
import re
import pickle
from collections import defaultdict
from collections import Counter
from multiprocessing import Pool

import argparse
import tqdm
from argparse import Namespace
from itertools import chain
from math import ceil
from src.model.generic import ModelWrapper
from src.model.generic import Modes
from src.model.all_models import load_all_models

from src.dataloaders.paths_bli import lang2bli_vocabulary
from src.model.text2vec import run_strip_accents

parser = argparse.ArgumentParser()
parser.add_argument("-lang", "--language", type=str, required=True)
parser.add_argument("-gpu", type=str, required=True)
parser.add_argument("-cf", "--context_frequency", type=int, required=True)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu



def create_index(raw_corpus, tgt_vocab_set, corpus_len):
  """
  Enumerates corpus and records for each word the document it appears in and the corresponding position (index), and
  record for each document the list of terms and their positions (doc2termlist)
  :param raw_corpus: list of documents
  :param tgt_vocab_set: vocabulary
  :param corpus_len: number of documents
  :return:
  """
  print("creating index")
  index = defaultdict(list)
  doc2termlist = {}
  # buffer = 20  # for out of position cases
  for did, sent in tqdm.tqdm(enumerate(raw_corpus), total=corpus_len):
    sent_tokens = sent.split()
    sent_set = set(sent_tokens)
    words_positions = []
    for word in sent_set:
      # skip if we have a non-zero target vocabulary and word is not contained
      if word not in tgt_vocab_set and tgt_vocab_set:
        continue
      position = sent_tokens.index(word)
      word_wordposition = (word, position)
      doc_wordposition = (did, position)
      words_positions.append(word_wordposition)
      index[word].append(doc_wordposition)
    doc2termlist[did] = words_positions

  notfound = {term for term in tgt_vocab_set if term not in index}
  print("No containing sentences found for %s terms" % len(notfound))
  print("context-count distribution:", end=" ")
  context_distr = {k: len(v) for k, v in index.items()}
  tmp = []
  for value in context_distr.values():
    if value < 5:
      tmp.append(value)
    elif 5 <= value < 10:
      tmp.append(10)
    elif 10 <= value < 20:
      tmp.append(20)
    elif 20 <= value < 30:
      tmp.append(30)
    elif 30 <= value < 40:
      tmp.append(40)
    elif 40 <= value < 50:
      tmp.append(50)
    elif value >= 50:
      tmp.append(60)
  print(Counter(tmp))
  return index, doc2termlist


def collect_embeddings(index, doc2termlist, tokenizer, init_model, raw_corpus, term2num_collected_embs, file_fqdn,
                       context_frequency):
  # debug line (term2doc): {term: [raw_corpus[docid] for docid, _ in doclist] for term, doclist in index.items()}
  selected_docs = set([sid for sid, _ in chain(*list(index.values()))])
  encoding_model = ModelWrapper(init_model())
  add_special_token = True

  # keep only docs that are associated with a vocabulary term
  print("effective corpus size: %s" % (str(len(selected_docs))))
  print("effective vocabulary size: %s" % str(len(index)))

  if os.path.exists(file_fqdn):
    print("\n\nWARNING: removing old results, starting from scratch\n\n")
    os.remove(file_fqdn)

  continue_appending = True if os.path.exists(file_fqdn) else False
  loaded_records = 0  # used as an offset

  if continue_appending:
    print("File already exists, finding offset to continue collecting embs")
    with open(file_fqdn, "rb") as f:
      try:
        while True:
          pickle.load(f)
          loaded_records += 1
      except EOFError:
        print("offset = %s docs" % str(loaded_records))
        pass

  special_token_adjustment = 1 if add_special_token else 0
  print("embedd corpus")
  skipped_documents = 0
  with open(file_fqdn, "wb+") as f:
    for did, doc in tqdm.tqdm(enumerate(raw_corpus), total=len(selected_docs)):
      if did not in selected_docs or did < loaded_records:
        continue

      sentences_masks = tokenizer.encode(doc, add_special_token=add_special_token, pad_to_max_length=False)
      words = sentences_masks[-1]

      # dry run to check if we need to run encoder (expensive)
      run_encoder = _dry_run(context_frequency, did, doc2termlist, special_token_adjustment, term2num_collected_embs,
                             words)

      if not run_encoder:
        continue

      # run encoder
      all_layers = encoding_model(sentences_masks=sentences_masks,
                                  word_aggr=Modes.ALL,
                                  wp_aggr=Modes.AVG,
                                  is_single_instance=True)[0]

      # embedding updates
      for term, position in doc2termlist[did]:
        position = position + special_token_adjustment
        if position < len(words) and term2num_collected_embs.get(term, 0) <= context_frequency:
          if term != words[position]:
            term_without_accents = run_strip_accents(term)
            # index shift can happens due to different tokenization types, e.g. "Fall's" -> ["Fall","\'s"] vs ["Fall's"]
            for i in range(10):
              # some hugginface models remove accents during tokenization (mbert-uncased)
              fw = position + i < len(words) and (term == words[position + i] or term_without_accents == words[position + i])
              if fw:
                position = position + i
                break
              bw = position - i >= 0 and (term == words[position - i] or term_without_accents == words[position - i])
              if bw:
                position = position - i
                break

          # might still evaluate to False because of </s> token, or different encodings of special characters
          if term == words[position] or term_without_accents == words[position]:
            # update_freqs:
            num_collected_embs = term2num_collected_embs.get(term, 0)
            num_collected_embs += 1
            term2num_collected_embs[term] = num_collected_embs
            # collect embedding for each layer
            for layer, emb_seq in enumerate(all_layers):
              pickle.dump((layer, term, emb_seq[position]), f)

  print("Skipped documents=%s" % str(skipped_documents))
  return term2num_collected_embs


def _dry_run(context_frequency, did, doc2termlist, special_token_adjustment, term2num_collected_embs, words):
  for term, position in doc2termlist[did]:
    position = position + special_token_adjustment
    if position < len(words) and term2num_collected_embs.get(term, 0) <= context_frequency:
      if term != words[position] and run_strip_accents(term) != words[position]:
        # index shift can happens due to different tokenization types, e.g. "Fall's" -> ["Fall","\'s"] vs ["Fall's"]
        for i in range(10):
          fw = position + i < len(words) and (term == words[position + i] or run_strip_accents(term) == words[position + i])
          bw = position - i >= 0 and (term == words[position - i] or run_strip_accents(term) == words[position - i])
          if fw or bw:
            return True
      else:
        return True
  return False


def chunks(lst, n):
  """Yield successive n-sized chunks from lst."""
  for i in range(0, len(lst), n):
    yield lst[i:i + n]


def main(config):
  config = Namespace(**config)

  MODELS = load_all_models()
  model_name, Model, Tokenizer = MODELS[5]

  lang = config.language
  context_frequency = config.context_frequency

  tokenizer = Tokenizer()
  do_lowercasing = False if tokenizer.is_cased else True
  cased_str = "lowercased" if do_lowercasing else "cased"
  basedir = "/path/to/AOC/%s_wikipedia_clean_full_vocab/AOC_cs=%s/" % (cased_str, str(context_frequency))

  basedir += "model=%s/" % model_name
  basedir += "%s/" % lang
  print(basedir)
  os.makedirs(basedir, exist_ok=True)
  os.makedirs(basedir + ".cache/", exist_ok=True)

  data_dir = "/path/to/wikipedia_dumps/%s_wiki_text/%s/clean_%s.txt" % (lang, lang, cased_str)
  # lang2casing2wiki_clean_size = {"cased": {"de": 10908734, "fi": 1415214, "ru": 6664301, "it": 7308595, "en": 38655285},
  #                                "lowercased": {"de": 7940373, "fi": 1415214, "ru": 6663447, "it": 7308964, "en": 38655558}}

  # sizes in terms of number of lines
  num_chunks = config.total_chunks
  lang2casing2wiki_clean_size = {"cased": {"de": 11365918, "fi": 1745173, "ru": 7484682, "it": 7414416, "en": 39096629},
                                 "lowercased": {"de": 11222425, "fi": 1742953, "ru": 7479080, "it": 7408792, "en": 38795859}}

  docidlist = list(range(lang2casing2wiki_clean_size[cased_str][lang]))
  chunked_docidlist = list(chunks(docidlist, n=ceil(lang2casing2wiki_clean_size[cased_str][lang] / num_chunks)))
  selected_chunk = chunked_docidlist[config.chunk]

  num_docs = len(selected_chunk)
  start = selected_chunk[0]
  end = selected_chunk[-1]

  tgt_vocab_set, type2vocab = get_vocab(cased_str, do_lowercasing, lang)

  #
  # clean Wikipedia
  #
  pattern = re.compile("\[\[\d+\]\]")
  print("dataloaders dir: %s" % data_dir)
  def get_clean_wiki_dataset(appendix=None):
    with open(data_dir) as f:
      for i, line in enumerate(f):
        line = line.strip()
        # line = clean(line, to_lower=do_lowercasing)
        if start <= i <= end:
          if line != "" and not re.match(pattern, line):
            yield line.strip()
          else:
            pass # empty line or line containing article number
        if i > end:
          break
        # if i > 0 and i % 1000 == 0:
        #   break
      if appendix is not None:
        for term in appendix:
          yield term

  cachefile = basedir + ".cache/term2num_collected_embs.pkl"
  if os.path.exists(cachefile):
    with open(cachefile, "rb") as f:
      term2num_collected_embs = pickle.load(f)
  else:
    term2num_collected_embs = {}

  index, doc2termlist = create_index(raw_corpus=get_clean_wiki_dataset(),
                                     corpus_len=num_docs,
                                     tgt_vocab_set=tgt_vocab_set)

  # ISO-backoff
  oov_terms = [term for term in tgt_vocab_set if term not in index]
  for term in oov_terms:
    docid = len(doc2termlist)
    position = 0
    doc2termlist[docid] = [(term, position)]
    index[term] = [(docid, position)]
  print("constructing embeddings for %s oov terms" % str(len(oov_terms)))

  # tokenizer, get_model, encode = get_model_spec(model_str=model_str, model=model)
  term2num_collected_embs = collect_embeddings(index=index,
                                               doc2termlist=doc2termlist,
                                               tokenizer=tokenizer,
                                               init_model=Model,
                                               raw_corpus=get_clean_wiki_dataset(oov_terms),
                                               term2num_collected_embs=term2num_collected_embs,
                                               file_fqdn=basedir + "layerTermEmb_tuples.pkl",
                                               context_frequency=context_frequency)
  with open(cachefile, "wb") as f:
    pickle.dump(term2num_collected_embs, f)


def get_vocab(cased_str, do_lowercasing, lang):
  tgt_vocab_set = set()
  clef_vocab_dir = "/path/to/vocabularies/%s/clef" % cased_str
  europarl_vocab_dir = clef_vocab_dir.replace("clef", "europarl")
  clef_vocab_path = clef_vocab_dir + "/corpusFasttext_vocab_intersection_%s.txt" % lang
  europarl_vocab_path = europarl_vocab_dir + "/corpusFasttext_vocab_intersection_%s.txt" % lang
  print("clef vocab: %s" % clef_vocab_path)
  with open(clef_vocab_path, "r") as f:
    clef_vocab = [line.strip() for line in f.readlines()]
  tgt_vocab_set.update(clef_vocab)
  europarl_vocab = []
  if lang != "ru":
    print("europarl vocab: %s" % europarl_vocab_path)
    with open(europarl_vocab_path, "r") as f:
      europarl_vocab = [line.strip() for line in f.readlines()]
  tgt_vocab_set.update(europarl_vocab)
  bli_vocab = lang2bli_vocabulary[lang]
  print("vocab file 1: %s" % bli_vocab)
  if lang in ["en", "de", "fi"]:
    idx = 0
  else:
    assert lang in ["it", "ru"]
    idx = 1
  with open(bli_vocab, "r") as f:
    lines = [line.strip() for line in f.readlines()]
  bli_vocab = [l.lower().split()[idx] for l in lines] if do_lowercasing else [l.lower().split()[idx] for l in lines]
  tgt_vocab_set.update(bli_vocab)
  return tgt_vocab_set, {"bli": bli_vocab, "clef": clef_vocab, "europarl": europarl_vocab}


if __name__ == '__main__':
  with Pool(processes=1) as pool:
    total_chunks = 1
    for current_chunk in range(total_chunks):
      print("Processing chunk %s out of %s:" % (str(current_chunk+1), str(total_chunks)))
      inpt = vars(args)
      inpt["chunk"] = current_chunk
      inpt["total_chunks"] = total_chunks
      pool.apply(main, args=(inpt,))
