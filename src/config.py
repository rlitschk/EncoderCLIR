# Set this variable to your LASER root directory, e.g. /home/usr/projects/LASER/
LASER_HOME = ""
LASER_EMB = LASER_HOME + "tasks/embed/embed.sh"

# optionally set data dir for nltk
NLTK_DATA = ""

import os
ROOT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + "/"
CLWE_BASE_DIR = os.path.join(ROOT_DIRECTORY, "data", "embedding_spaces")
CKPT_DIR = os.path.join(ROOT_DIRECTORY, "data", "checkpoints") + "/"
OUTPUT_DIR = os.path.join(ROOT_DIRECTORY, "results") + "/"

#
# Model checkpoints
#
HF_XLM_TAG = "xlm-mlm-100-1280" # options: xlm-roberta-large, xlm-roberta-base
MBERT_UNCASED_TAG = "bert-base-multilingual-uncased" # options: bert-base-multilingual-uncased, bert-base-multilingual-cased, distilbert-base-multilingual-cased
DISTIL_MDISTILLBERT_TAG = "distilbert-multilingual-nli-stsb-quora-ranking"
DISTIL_XLMR_TAG = "xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
DISTIL_mUSE_TAG = "distiluse-base-multilingual-cased"
DISTIL_TAGS = {DISTIL_mUSE_TAG, DISTIL_XLMR_TAG, DISTIL_MDISTILLBERT_TAG}
distilmodel2path = {
  DISTIL_MDISTILLBERT_TAG: CKPT_DIR + "torch/sentence_transformers/public.ukp.informatik.tu-darmstadt.de_reimers_sentence-transformers_v0.2_distilbert-multilingual-nli-stsb-quora-ranking.zip/",
  DISTIL_mUSE_TAG: CKPT_DIR + "torch/sentence_transformers/sbert.net_models_distiluse-base-multilingual-cased/",
  DISTIL_XLMR_TAG: CKPT_DIR + "torch/sentence_transformers/public.ukp.informatik.tu-darmstadt.de_reimers_sentence-transformers_v0.2_xlm-r-100langs-bert-base-nli-stsb-mean-tokens.zip/"}
HUGGINGFACE_CACHE_DIR = CKPT_DIR + "huggingface/"
TF_HUB_DIR = CKPT_DIR + "tf_hub/"


# Optional: Download and uppack in PATH_PHANTOM_JS https://phantomjs.org/download.html
PATH_PHANTOM_JS = "/home/usr/bin/phantomjs-2.1.1-linux-x86_64/bin/phantomjs"

# Directory from where to store/load google translated queries
GTRANSLATE_CACHE = ROOT_DIRECTORY + "data/gtranslated_clef_queries/"
PROCRUSTES_PATH = OUTPUT_DIR + "procrustes/"

# download (https://github.com/codogogo/xling-eval/tree/master/bli_datasets) and set variable
BLI_DICTS_PATH = "/work/usr/data/word_embs/yacle/translations/freq_split/pairwise/"

