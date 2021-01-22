import torch

from src import config as c
from src.model.text2vec import lookup
from functools import partial

class HF_mBERT_Model:
  """
  bert-base-multilingual-cased -> 12 + 1 layers
  """
  def __init__(self, model_specifier_path=c.MBERT_UNCASED_TAG, cache_dir=c.HUGGINGFACE_CACHE_DIR):
    from transformers import BertModel
    with torch.no_grad():
      self.model = BertModel.from_pretrained(model_specifier_path,
                                             cache_dir=cache_dir,
                                             output_hidden_states=True)
    self.model.eval()
    self.model.output_hidden_states = True
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    self.model.to(self.device, non_blocking=True)
    self.model.output_hidden_states = True

  def __call__(self, word_ids, lengths, langs=None, transform_input=False):
    if transform_input:
      raise NotImplementedError("transform input not yet implemented for huggingface models")

    max_len = torch.tensor(word_ids.shape[0])
    attention_mask = (torch.arange(max_len).expand(len(lengths.cpu()), max_len) < lengths.cpu().unsqueeze(1)).int()
    # attention_mask = torch.transpose(attention_mask, 1, 0)
    attention_mask = attention_mask.cuda()
    word_ids = torch.transpose(word_ids, 1, 0)
    with torch.no_grad():
      _, _, all_layers = self.model(word_ids, encoder_hidden_states=True, attention_mask=attention_mask)
    # all_layers = [torch.transpose(l, 0, 1) for l in all_layers]
    return all_layers

import unicodedata
def run_strip_accents(text):
  """
  Strips accents from a piece of text.
  """
  text = unicodedata.normalize("NFD", text)
  output = []
  for char in text:
    cat = unicodedata.category(char)
    if cat == "Mn":
      continue
    output.append(char)
  return "".join(output)


class HF_mBERT_Tokenizer:

  def __init__(self, model_specifier_path=c.MBERT_UNCASED_TAG, cache_dir=c.HUGGINGFACE_CACHE_DIR):
    from transformers import BertTokenizer
    self.tokenizer = BertTokenizer.from_pretrained(model_specifier_path, cache_dir=cache_dir)
    # self.is_cased = False if c.MBERT_TAG == 'bert-base-multilingual-uncased' else True
    self.is_cased = self.tokenizer.encode("Hallo") != self.tokenizer.encode("hallo")
    self.BOS_ID = self.tokenizer.cls_token_id
    self.BOS = self.tokenizer.convert_ids_to_tokens(self.BOS_ID)
    self.PAD_ID = self.tokenizer.pad_token_id
    self.EOW_TOKEN = "##"
    self.SEP_TOKEN = '[SEP]'
    self.UNK_TOKEN = '[UNK]'
    self.SEP_ID = self.tokenizer.convert_tokens_to_ids(self.SEP_TOKEN)

  def tokenize(self, txt):
    return self.tokenizer.tokenize(txt)

  def encode(self, txt, add_special_token=False, max_length=128, pad_to_max_length=True,
             keepwords=None, vocab2idf=None):
    wordpiece_tokens = []
    reconstruction_mask = []
    word_tokens = []
    idfs = []

    offset = 2
    _max = max_length -offset if add_special_token else max_length
    include_all = True if max_length == -offset else False

    configured_lookup = partial(lookup, language="tmp", embedding_lookup={"tmp": vocab2idf})

    if not self.is_cased:
      txt = txt.lower()

    word_pieces = []
    for word_piece in self.tokenize(txt):
      is_first_wordpiece = not word_piece.startswith(self.EOW_TOKEN)

      if is_first_wordpiece:
        if len(word_pieces) == 0:
          word_pieces.append(word_piece)
        else:
          if len(wordpiece_tokens) + len(word_pieces) < _max or include_all:
          # if (len(word_tokens) + 1) < _max or include_all:
            word = "".join([wp.replace("##","") for wp in word_pieces])
            if (keepwords and word.lower() in keepwords) or keepwords is None:
              wordpiece_tokens.extend(word_pieces)
              reconstruction_mask.append(len(word_pieces))
              word_tokens.append(word)
              if vocab2idf:
                # normalized_word, idf = configured_lookup(word)
                # idf = idf if idf else 0
                # idfs.append(idf)
                idf = vocab2idf.get(word.lower(), 0)
                idfs.append(idf)
            word_pieces = [word_piece]
          else:
            break
      else:
        word_pieces.append(word_piece)

    if word_pieces and len(wordpiece_tokens) + len(word_pieces) < _max or include_all:
      wordpiece_tokens.extend(word_pieces)
      reconstruction_mask.append(len(word_pieces))
      word = "".join([wp.replace("##","") for wp in word_pieces])
      word_tokens.append(word)
      if vocab2idf:
        # normalized_word, idf = configured_lookup(word)
        # idf = idf if idf else 0
        # idfs.append(idf)
        idf = vocab2idf.get(word.lower(), 0)
        idfs.append(idf)

    sent_tokens = [self.tokenizer.convert_tokens_to_ids(token) for token in wordpiece_tokens]
    if add_special_token:
      sent_tokens = [self.BOS_ID] + sent_tokens
      reconstruction_mask = [1] + reconstruction_mask
      word_tokens = [self.BOS] + word_tokens
      mean_idf = sum(idfs) / max([len(idfs), 1])

      if vocab2idf:
        idfs = [mean_idf] + idfs

      if offset == 2:
        word_tokens = word_tokens + [self.SEP_TOKEN]
        sent_tokens += [self.SEP_ID]
        reconstruction_mask += [1]
        if vocab2idf:
          idfs = idfs + [mean_idf]

    sequence_length = torch.tensor(len(sent_tokens))
    if pad_to_max_length and len(sent_tokens) < max_length and not include_all:
      difference = max_length - len(sent_tokens)
      sent_tokens = sent_tokens + [self.PAD_ID] * difference
    sent = torch.LongTensor(sent_tokens)
    if vocab2idf:
      assert len(idfs) == len(reconstruction_mask)
    return sent, sequence_length, reconstruction_mask, idfs, word_tokens



