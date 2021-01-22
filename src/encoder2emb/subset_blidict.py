import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import random
import tqdm
import pickle
from functools import partial
from src import config as c
from src.dataloaders.extractors import load_txt_data
from src.model.generic import ModelWrapper
from src.model.bytepairencoding import InteractiveBPE
from src.model.generic import Modes

# creates embedding table, later used to create X_L1 and X_L2 for SVD (procrustes)
def subset_blidict(src_lang, tgt_lang, avg_layers, single_token_words_only=False):
  if single_token_words_only:
    base_dir = c.PROCRUSTES_PATH + "STO_avg_layers=%s_%s-%s/" % (str(avg_layers), src_lang, tgt_lang)
  else:
    base_dir = c.PROCRUSTES_PATH + "avg_layers=%s_%s-%s/" % (str(avg_layers), src_lang, tgt_lang)

  os.makedirs(base_dir, exist_ok=True)

  # output files
  path_emb_table = base_dir + "embs.npy"
  path_vocab = base_dir + "bli_vocabulary.txt"
  path_train = base_dir + "train.txt"
  # path_dev = base_dir + "dev.txt"

  os.makedirs(base_dir, exist_ok=True)
  inputfile = c.BLI_DICTS_PATH + "%s-%s/yacle.train.freq.5k.%s-%s.tsv" % (src_lang, tgt_lang, src_lang, tgt_lang)
  if os.path.exists(inputfile):
    all_lines = load_txt_data(inputfile)
  else:
    inputfile = c.BLI_DICTS_PATH + "%s-%s/yacle.train.freq.5k.%s-%s.tsv" % (tgt_lang, src_lang, tgt_lang, src_lang)
    assert os.path.exists(inputfile)
    all_lines = load_txt_data(inputfile)

  # num_train = 5000
  # num_dev = len(all_lines) - num_train

  random.seed(0)
  random.seed(432)
  random_permutation = list(range(len(all_lines)))
  random.shuffle(random_permutation)
  # tgt_indices = random_permutation[-num_dev:]
  # src_indices = random_permutation[:num_train]

  # write down vocabulary file
  all_tokens = []
  bpe = InteractiveBPE()
  bpe.open()
  indices = []

  for idx, line in enumerate(all_lines):
    if single_token_words_only:
      tokens = line.split()
      tgt = bpe.word2bpe(tokens[1])
      src = bpe.word2bpe(tokens[0])
      if len(src.split()) == 1 and len(tgt.split()) == 1:
        all_tokens.extend(tokens)
        indices.append(idx)
    else:
      all_tokens.extend(line.split())
      indices.append(idx)
  bpe.close()
  unique_tokens = set(all_tokens)

  id2word = {i: token for i, token in enumerate(unique_tokens)}
  word2id = {v:k for k, v in id2word.items()}
  assert len(id2word) == len(word2id)
  with open(path_vocab, "wb") as f:
    pickle.dump(id2word, f)

  # write down train and dev instances, words replaced by IDs
  train_instances = []
  for idx in indices:
    word1, word2 = all_lines[idx].split()
    train_instances.append((word2id[word1], word2id[word2]))
  # dev_instances = []
  # for idx in tgt_indices:
  #   word1, word2 = all_lines[idx].split()
  #   dev_instances.append((word2id[word1], word2id[word2]))
  with open(path_train, "wb") as f:
    pickle.dump(train_instances, f)
  # with open(path_dev, "wb") as f:
  #   pickle.dump(dev_instances, f)

  # serialize embedding table
  encoder = ModelWrapper()
  emb_initializer = []
  cache = []
  mask_cache = []
  idf_cache = []
  batchsize = 100
  interactiveBPE = InteractiveBPE()
  interactiveBPE.open()
  encode_fn = partial(encoder.__call__,
                      wp_aggr=Modes.AVG,
                      sent_aggr=Modes.ALL,
                      add_sent_delimiter=False,
                      avg_layers=avg_layers)
  for i in tqdm.tqdm(range(len(word2id))):
    word = id2word[i]
    bpe = interactiveBPE.word2bpe(word)
    cache.append(bpe)
    mask = [len(bpe.split())]
    mask_cache.append(mask)
    idf = [1.0 for _ in range(len(mask))] # dummy idf
    idf_cache.append(idf)
    if len(cache) == batchsize:
      emb = encode_fn((cache, mask_cache, idf_cache))
      emb_initializer.extend(emb)
      cache = []
      mask_cache = []
      idf_cache = []

  if len(cache) > 0:
    emb_initializer.extend(encode_fn((cache, mask_cache, idf_cache)))
  emb_initializer = np.array(emb_initializer, dtype=np.float32)
  with open(path_emb_table, "wb") as f:
    np.save(f, emb_initializer)
  print("Done with %s-%s" % (src_lang, tgt_lang))


onetoken_procrustes = False
for avgl in [0, -4]:
  subset_blidict("en", "de", avgl, onetoken_procrustes)
  subset_blidict("en", "fi", avgl, onetoken_procrustes)
  subset_blidict("en", "it", avgl, onetoken_procrustes)
  subset_blidict("en", "ru", avgl, onetoken_procrustes)
  subset_blidict("de", "fi", avgl, onetoken_procrustes)
  subset_blidict("de", "it", avgl, onetoken_procrustes)
  subset_blidict("de", "ru", avgl, onetoken_procrustes)
  subset_blidict("fi", "it", avgl, onetoken_procrustes)
  subset_blidict("fi", "ru", avgl, onetoken_procrustes)
