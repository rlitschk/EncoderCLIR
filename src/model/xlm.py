import torch
from functools import partial
from src.model.text2vec import lookup
from src import config as c


class HF_XLM_Model:
  def __init__(self, pretrained_model_name_or_path):
    from transformers import XLMModel
    with torch.no_grad():
      self.model = XLMModel.from_pretrained(pretrained_model_name_or_path,
                                            cache_dir=c.HUGGINGFACE_CACHE_DIR,
                                            output_hidden_states=True)
    self.model.eval()
    self.model.output_hidden_states = True
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    self.model.to(self.device, non_blocking=True)

  def __call__(self, word_ids, lengths, langs=None, transform_input=False):
    if transform_input:
      raise NotImplementedError("transform input not yet implemented for huggingface models")
    with torch.no_grad():
      _, all_layers = self.model(torch.transpose(word_ids, 0, 1), lengths=lengths)
    # all_layers = [torch.transpose(l, 0, 1) for l in all_layers]
    return all_layers


class HF_XLM_Tokenizer:

  def __init__(self, pretrained_model_name_or_path=c.HF_XLM_TAG, cache_dir=c.HUGGINGFACE_CACHE_DIR):
    from transformers import XLMTokenizer
    self.tokenizer = XLMTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir)
    do_lowercase_and_remove_accent = \
      self.tokenizer.pretrained_init_configuration[c.HF_XLM_TAG]["do_lowercase_and_remove_accent"]
    # len([elem for elem in list(self.tokenizer.encoder.keys()) if elem == elem.lower()]) == len(self.tokenizer.encoder.keys())
    self.is_cased = not do_lowercase_and_remove_accent
    if self.is_cased:
      assert self.tokenizer.encode("Hallo") != self.tokenizer.encode("hallo")
    self.PAD_TOKEN = "<pad>"
    self.BOS_TOKEN = "<s>"
    self.EOS_TOKEN = "</s>"
    self.EOW_TOKEN = "</w>"
    self.UNK_TOKEN = "<unk>"

  def tokenize(self, txt):
    return self.tokenizer.tokenize(txt)

  def encode(self, txt, add_special_token=False, max_length=128, pad_to_max_length=True,
             keepwords=None, vocab2idf=None):
    if keepwords is not None:
      raise NotImplementedError()

    word_tokens = []
    idfs = []
    worpiece_tokens = []
    reconstruction_mask = []

    _max = max_length - 2 if add_special_token else max_length
    include_all = True if max_length == -1 else False
    configured_lookup = partial(lookup, language="tmp", embedding_lookup={"tmp": vocab2idf})

    word_pieces = []
    for word_piece in self.tokenize(txt):
      word_pieces.append(word_piece)
      is_last_wordpiece = word_piece.endswith(self.EOW_TOKEN)
      if is_last_wordpiece:
        if (len(worpiece_tokens) + len(word_pieces)) < _max or include_all:
        # if (len(word_tokens) + 1) < _max or include_all:
          worpiece_tokens.extend(word_pieces)
          reconstruction_mask.append(len(word_pieces))
          word = "".join(word_pieces).replace(self.EOW_TOKEN, "")
          word_tokens.append(word)
          if vocab2idf:
            # normalized_word, idf = configured_lookup(word)
            # idf = idf if idf else 0
            idf = vocab2idf.get(word.lower(), 0)
            idfs.append(idf)
          word_pieces = []
        else:
          break

    sent = [self.tokenizer.convert_tokens_to_ids(token) for token in worpiece_tokens]
    if add_special_token:
      sent = [self.tokenizer.convert_tokens_to_ids(self.BOS_TOKEN)] + sent + [self.tokenizer.convert_tokens_to_ids(self.EOS_TOKEN)]
      reconstruction_mask = [1] + reconstruction_mask + [1]
      word_tokens = [self.BOS_TOKEN] + word_tokens + [self.EOS_TOKEN]
      if vocab2idf:
        mean_idf = sum(idfs) / max([len(idfs),1])
        idfs = [mean_idf] + idfs + [mean_idf]

    sequence_length = torch.tensor(len(sent))
    if pad_to_max_length and len(sent) < max_length and not include_all:
      difference = max_length - len(sent)
      sent = sent + [self.tokenizer.convert_tokens_to_ids(self.PAD_TOKEN)] * difference
    sent = torch.LongTensor(sent)
    return sent, sequence_length, reconstruction_mask, idfs, word_tokens
