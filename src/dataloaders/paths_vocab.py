from src.config import ROOT_DIRECTORY

#
# Vocabulary paths (use same vocabulary for xlm, mbert as for procB)
#
fasttext_basedir = ROOT_DIRECTORY + "procb/"
lang2fasttext_vocab = {
  "en": fasttext_basedir + "en-it/vocab_en-it.en.yacle.train.freq.5k.pkl",
  "fi": fasttext_basedir + "en-fi/vocab_en-fi.fi.yacle.train.freq.5k.pkl",
  "ru": fasttext_basedir + "en-ru/vocab_en-ru.ru.yacle.train.freq.5k.pkl",
  "it": fasttext_basedir + "en-it/vocab_en-it.it.yacle.train.freq.5k.pkl",
  "de": fasttext_basedir + "de-fi/vocab_de-fi.de.yacle.train.freq.5k.pkl"
}