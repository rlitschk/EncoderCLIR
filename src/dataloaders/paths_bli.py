#
# BLI dictionary paths
#

# path to bli dicts, download here: https://github.com/codogogo/xling-eval/tree/master/bli_datasets
bli_basedir = "/work/usr/data/word_embs/yacle/translations/freq_split/pairwise/"
lang2bli_vocabulary = {
  "en": bli_basedir + "en-fi/yacle.train.freq.5k.en-fi.tsv",
  "de": bli_basedir + "de-fi/yacle.train.freq.5k.de-fi.tsv",
  "fi": bli_basedir + "fi-it/yacle.train.freq.5k.fi-it.tsv",
  "ru": bli_basedir + "en-ru/yacle.train.freq.5k.en-ru.tsv",
  "it": bli_basedir + "en-it/yacle.train.freq.5k.en-it.tsv"
}
lang_pair2bli_dict = {
  ("en", "fi"): bli_basedir + "en-fi/yacle.train.freq.5k.en-fi.tsv",
  ("en", "it"): bli_basedir + "en-it/yacle.train.freq.5k.en-it.tsv",
  ("en", "ru"): bli_basedir + "en-ru/yacle.train.freq.5k.en-ru.tsv",
  ("en", "de"): bli_basedir + "en-de/yacle.train.freq.5k.en-de.tsv",
  ("de", "fi"): bli_basedir + "de-fi/yacle.train.freq.5k.de-fi.tsv",
  ("de", "it"): bli_basedir + "de-it/yacle.train.freq.5k.de-it.tsv",
  ("de", "ru"): bli_basedir + "de-ru/yacle.train.freq.5k.de-ru.tsv",
  ("fi", "it"): bli_basedir + "fi-it/yacle.train.freq.5k.fi-it.tsv",
  ("fi", "ru"): bli_basedir + "fi-ru/yacle.train.freq.5k.fi-ru.tsv",
}