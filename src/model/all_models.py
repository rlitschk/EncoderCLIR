from src.config import *

def load_all_models():
  """
  """
  from src.model.xlm import HF_XLM_Model
  from src.model.xlm import HF_XLM_Tokenizer
  from src.model.mbert import HF_mBERT_Model
  from src.model.mbert import HF_mBERT_Tokenizer

  XLM_HF_MODEL = (
    "hf_" + HF_XLM_TAG,
    lambda: HF_XLM_Model(pretrained_model_name_or_path=HF_XLM_TAG),
    lambda: HF_XLM_Tokenizer(pretrained_model_name_or_path=HF_XLM_TAG)
  )

  mBERT_UNCASED_HF_MODEL = (
    "hf_" + MBERT_UNCASED_TAG,
    lambda: HF_mBERT_Model(model_specifier_path=MBERT_UNCASED_TAG),
    lambda: HF_mBERT_Tokenizer(model_specifier_path=MBERT_UNCASED_TAG)
  )

  DISTIL_DISTILLmBERT_MODEL = (
    DISTIL_MDISTILLBERT_TAG,
    DISTIL_MDISTILLBERT_TAG,
    None # lambda: DistillMBERTTokenizer(pretrained_model_name_or_path=DISTIL_MDISTILLBERT_TAG)
  )

  DISTIL_XLM_R_MODEL = (
    DISTIL_XLMR_TAG,
    DISTIL_XLMR_TAG,
    None # lambda: HF_XLM_R_Tokenizer(pretrained_model_name_or_path=DISTIL_XLMR_TAG, is_sbert_model=True)
  )

  DISTIL_USE_MODEL = (
    DISTIL_mUSE_TAG,
    DISTIL_mUSE_TAG,
    None # lambda: HF_XLM_R_Tokenizer(pretrained_model_name_or_path=DISTIL_XLMR_TAG, is_sbert_model=True)
  )

  LASER_MODEL = (
    #LASER,
    #LASER,
    "laser",
    "laser",
    None #lambda: CustomLASERTokenizer(codes=LASER_BPE_CODES, vocab=LASER_BPE_VOCAB)
  )

  LABSE_MODEL = ( "labse", "labse", "labse" )
  MUSE_MODEL = ( "muse", "muse", "muse")

  MODELS = {
    # XLM
    "xlm": XLM_HF_MODEL, # 0
    # mBERT
    "mbert": mBERT_UNCASED_HF_MODEL, # 1
    # SBERT
    "distil_mbert": DISTIL_DISTILLmBERT_MODEL, # 2
    "distil_xlmr": DISTIL_XLM_R_MODEL, # 3
    "distil_muse": DISTIL_USE_MODEL, # 4
    # Other
    "laser": LASER_MODEL, # 5
    "labse": LABSE_MODEL, # 6
    "muse": MUSE_MODEL
  }
  return MODELS


def load_unspecialized_encoders():
  return load_all_models()[:2]