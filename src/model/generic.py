import torch

from src.model.aggregation import aggregate_torch
from src.model.aggregation import Modes


class ModelWrapper:
  """
  The job of this class is to run the internal model and apply uniform pre- and post-processing
  """

  def __init__(self, model):
    # store ISO/AOC embedding table
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    self.model = model
    self.lang2term2weight = None
    self.lang2term2emb = None
    self.apply_backoff = False

  def set_weights(self, lang2term2weight):
    self.lang2term2weight = lang2term2weight

  def set_embeddings(self, lang2term2emb):
    cuda_lang2term2emb = {}
    for lang, term2emb in lang2term2emb.items():
      cuda_term2emb = {}
      for term, emb in term2emb.items():
        cuda_term2emb[term] = torch.from_numpy(emb).cuda()
      cuda_lang2term2emb[lang] = cuda_term2emb
    self.lang2term2emb = cuda_lang2term2emb

  def setup_backoff(self, lang2term2weight, lang2term2emb):
    self.apply_backoff = True
    self.set_weights(lang2term2weight)
    self.set_embeddings(lang2term2emb)

  @staticmethod
  def _reconstruct_words(mask, mode, seq):
    total_word_count = len(mask)
    word_embs = torch.stack([aggregate_torch(word_emb_components, mode) for word_emb_components in torch.split(seq, mask)])
    assert (total_word_count == len(word_embs) and mode != Modes.ALL) or mode == Modes.ALL
    return word_embs

  def _combine_with_static_embs(self, word_embs, words, language):
    tmp = []
    if language not in self.lang2term2emb: # or language not in self.lang2term2weight:
      return word_embs

    assert len(words) == len(word_embs)
    term2weight = self.lang2term2weight[language]
    term2emb = self.lang2term2emb[language]
    for i, word in enumerate(words):
      dynamic_emb = word_embs[i]
      if word is not None and (word in term2emb or word.lower() in term2emb):
        static_emb = term2emb[word] if word in term2emb else term2emb[word.lower()]
        weight = term2weight[word] if word in term2weight else term2weight[word.lower()]
        tmp.append(weight * static_emb + (1 - weight) * dynamic_emb)
      else:
        tmp.append(dynamic_emb)
    return tmp

  def generate_sentence_embeddings(self,
                                   embedded_sequece,
                                   masks,
                                   idfs,
                                   lengths,
                                   language,
                                   word_seqs=None,
                                   word_aggr=Modes.AVG,
                                   wp_aggr=Modes.AVG,
                                   normalize_word_embs=False):
    """
    Transforms (batch) sequence of word-piece embeddings into sentence embeddings.
    :param embedded_sequece: expected shape = [batchsize x seq-len x hidden size]
    :param masks: sequence masks = [batchsize x seq-len]
    :param idfs: idf sequences
    :param lengths: sequence lengths
    :param language: @deprecated
    :param word_seqs: @deprecated (old version used to combine SEMB with static CLWEs, words used for emb lookup)
    :param word_aggr: aggregation function applied on words
    :param wp_aggr: aggregation function applied on word-pieces
    :param normalize_word_embs: whether to apply l2-normalization
    :return:
    """
    # batchsize x seq-len x hidden size
    assert len(embedded_sequece.shape) == 3
    instances = []
    num_instances = len(masks)
    for instance_id in range(num_instances):
      instance_length = lengths[instance_id]
      word_embs = embedded_sequece[instance_id, :,:][:instance_length]
      mask = masks[instance_id]
      num_words = len(mask)
      num_tokens = sum(mask)
      words = word_seqs[instance_id]

      # aggregate word-piece embeddings into word embeddings if we don't want all WP's individually
      if wp_aggr != Modes.ALL:
        word_embs = ModelWrapper._reconstruct_words(mask=mask, mode=wp_aggr, seq=word_embs)

      # aggregate current embeddings with provided static (AOC/ISO) embeddings
      if words and self.lang2term2emb and language in self.lang2term2emb and language in self.lang2term2weight:
        tmp = self._combine_with_static_embs(word_embs, words, language)
        word_embs = torch.stack(tmp)

      # apply IDF scaling, disable when we use only [CLS]/<s> representations
      if idfs and not (word_aggr == Modes.FIRST and wp_aggr == Modes.FIRST):
        if idfs[instance_id] and wp_aggr is not Modes.ALL:
          assert len(idfs[instance_id]) == len(word_embs)
          if normalize_word_embs:
            word_embs = [emb/torch.norm(emb) for emb in word_embs]
          word_embs = torch.stack([emb * idf for idf, emb in zip(idfs[instance_id], word_embs)])

      # aggregate word embeddings into sentence embeddings
      instances.append(aggregate_torch(word_embs, mode=word_aggr, length=num_words if wp_aggr is not Modes.ALL else num_tokens))

    if word_aggr == Modes.ALL:
      return instances
    else:
      return torch.stack(instances)


  def __call__(self,
               sentences_masks,
               word_aggr,
               wp_aggr,
               avg_layers=None,  # -4 -> aggregate last four layers, 4 -> aggregate first four layers, 0 -> emb-layer only
               language=None,
               apply_procrustes=False,
               is_single_instance=False):
    """
    Runs embedding model and aggregates wp-embeddings into word-embeddings and word embeddings into sentence embeddings.
    :param sentences_masks: mask to zero-out padding positions.
    :param word_aggr: aggregation function applied on words
    :param wp_aggr: aggregation function applied on word-pieces
    :param avg_layers: @deprecard
    :param language: @deprecated
    :param apply_procrustes: @deprecated
    :param is_single_instance: flag whether input consists of single instance
    :return:
    """

    word_ids, lengths, masks, idfs, words = sentences_masks
    batch_size = len(word_ids)
    # assert len(lengths) == len(masks) == len(idfs) == len(words)

    if is_single_instance:
      batch_size = 1
      # handle single-instance input
      if word_ids.ndim == 1:
        word_ids = [word_ids]
      if lengths.ndim == 0:
        lengths = torch.unsqueeze(lengths, 0)
      if type(masks[0]) == int:
        masks = [masks]

    # reshape input for model
    # all_equal_len = all([list(word_ids[1].shape)[0] == list(elem.shape)[0] for elem in word_ids])
    if type(word_ids) == list or type(word_ids) == tuple: # and all_equal_len:
      word_ids = torch.stack(word_ids)
      word_ids = torch.transpose(word_ids, 1, 0)
    if type(lengths) == list or type(lengths) == tuple:
      lengths = torch.stack(lengths)

    # Encoding
    lengths = lengths.to(self.device)
    word_ids = word_ids.to(self.device)
    # layers x batch_size x seq_len x hidden_size
    all_layers = self.model(word_ids=word_ids,
                            lengths=lengths,
                            langs=None,
                            transform_input=False)

    # selected_layers = all_layers[avg_layers:] if avg_layers < 0 else all_layers[:avg_layers]
    # output_layer = torch.mean(torch.stack(selected_layers), dim=0)
    # output_layer = all_layers[avg_layers]

    # batch_size x seq_len x hidden_size
    if type(all_layers) != list and type(all_layers) != tuple:
      if len(all_layers.size()) == 3:
        all_layers = torch.unsqueeze(all_layers, 0)
      # batch_size x hidden_size
      elif len(all_layers.size()) == 2:
        return all_layers.detach().cpu().numpy()

    all_layers_sentembs = [self.generate_sentence_embeddings(embedded_sequece=output_layer, masks=masks,
                                                             idfs=idfs, word_aggr=word_aggr, wp_aggr=wp_aggr,
                                                             lengths=lengths, word_seqs=words, language=language)
                           for output_layer in all_layers]

    # torch.cuda.empty_cache()
    representations = []
    for i in range(batch_size):
      collected_layer_embs = [layer_sent_emb[i].cpu().detach().numpy() for layer_sent_emb in all_layers_sentembs]
      representations.append(collected_layer_embs)

    return representations
