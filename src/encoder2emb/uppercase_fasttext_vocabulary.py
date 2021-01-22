"""
Some BLI dicts contain only lowercased entries while some models (e.g. XLM) work on cased data, this script re-introduces
casing in BLI dicts.
"""
from src.dataloaders.paths_vocab import lang2fasttext_vocab
import pickle


def intersect(lowercased_fasttextvocab2id, corpus_vocab2freq, do_lowercased):
  fixed = []
  fixed_vocab2freq = {}
  for term_lower in lowercased_fasttextvocab2id.keys():
    term_upper = term_lower[0].upper() + term_lower[1:]
    if do_lowercased:
      term_upper = term_lower

    if term_lower in fixed_vocab2freq or term_upper in fixed_vocab2freq:
      continue

    lowercase_freq = corpus_vocab2freq.get(term_lower, 0)
    uppercase_freq = corpus_vocab2freq.get(term_upper, 0)

    if lowercase_freq == 0 and uppercase_freq == 0:
      continue #out of vocabulary case
    elif uppercase_freq > lowercase_freq:
      fixed.append(term_upper)
      assert term_upper not in fixed_vocab2freq
      fixed_vocab2freq[term_upper] = uppercase_freq
    else:
      # in doubt use lowercased version, covers many anglicism cases
      fixed.append(term_lower)
      assert term_lower not in fixed_vocab2freq
      fixed_vocab2freq[term_lower] = lowercase_freq
  return fixed, fixed_vocab2freq


corpus_languages = []#"de", "fi", "it", "ru"]
query_languages = ["en"]

for casing in ["cased", "lowercased"]:
  for dataset  in ["clef", "europarl"]:
    print("running %s" % dataset)
    for lang in corpus_languages + query_languages:
      if lang == "ru" and dataset == "europarl":
        continue

      print(lang)
      with open(lang2fasttext_vocab[lang], "rb") as f:
        fasttext_vocab2id = pickle.load(f)
      base_dir = "/work/user/vocabularies/%s/%s/" % (casing, dataset)

      filename = (base_dir + "full/%s_doc_vocab2freq.pkl") % lang
      if lang == "en":
        filename = filename.replace("doc", "query")

      with open(filename, "rb") as f:
        corpus_vocab2freq = pickle.load(f)


      cased_corpus_fasttext_intersection_vocab, vocab2freq =\
        intersect(lowercased_fasttextvocab2id=fasttext_vocab2id,
                  corpus_vocab2freq=corpus_vocab2freq,
                  do_lowercased= casing == "lowercased")

      with open(base_dir + "corpusFasttext_vocab_intersection_%s.txt" % lang, "w") as f:
        f.writelines([term + "\n" for term in cased_corpus_fasttext_intersection_vocab])
      with open(base_dir + "vocab2freq_%s.pkl" % lang, "wb") as f:
        pickle.dump(vocab2freq, f)
      with open(base_dir + "vocab2id_%s.pkl" % lang, "wb") as f:
        pickle.dump({term: _id for _id, term in enumerate(cased_corpus_fasttext_intersection_vocab)}, f)
