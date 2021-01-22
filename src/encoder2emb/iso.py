import os
import numpy as np
import tqdm
import argparse

from collections import defaultdict
from src.model.all_models import load_unspecialized_encoders
from src.model.generic import ModelWrapper
from src.model.generic import Modes

parser = argparse.ArgumentParser()
parser.add_argument("-gpu", type=str)
# parser.add_argument("-multiling", "--use_multiling_enc", type=lambda string: True if string == "True" else False)
# parser.add_argument("-fasttextonly", "--use_only_fasttext_vocab", type=lambda string: True if string == "True" else False, default=False)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def run(basedir, lang, encoding_model, tokenizer, add_special_token):
  tgt_file_0 = basedir + "%s_layer_%s.npy" % (lang, str(0))
  if os.path.exists(tgt_file_0):
    print("%s file exists already." % tgt_file_0)
    return

  tgt_vocab_set = set()

  # bli_vocab = lang2bli_vocabulary[lang]
  # print("vocab file 1: %s" % bli_vocab)
  # if lang in ["en", "de", "fi"]:
  #   idx = 0
  # else:
  #   assert lang in ["it", "ru"]
  #   idx = 1
  # with open(bli_vocab, "r") as f:
  #   lines = [line.strip() for line in f.readlines()]
  # bli_vocab = [l.split()[idx] for l in lines] if tokenizer.is_cased else [l.lower().split()[idx] for l in lines]
  # tgt_vocab_set.update(bli_vocab)

  casing = "cased" if tokenizer.is_cased else "lowercased"
  vocab_base_dir = "/work/usr/vocabularies/"
  vocab_path_clef = vocab_base_dir + "%s/%s/corpusFasttext_vocab_intersection_%s.txt" % (casing, "clef", lang)
  print("vocab file 2: %s" % vocab_path_clef)
  with open(vocab_path_clef, "r") as f:
    clef_vocab = [line.strip() for line in f.readlines()]
  tgt_vocab_set.update(clef_vocab)

  if lang != "ru":
    vocab_path_europarl = vocab_base_dir + "%s/%s/corpusFasttext_vocab_intersection_%s.txt" % (casing, "europarl", lang)
    print("vocab file 3: %s" % vocab_path_europarl)
    with open(vocab_path_europarl, "r") as f:
      europarl_vocab = [line.strip() for line in f.readlines()]
    tgt_vocab_set.update(europarl_vocab)

  vocabulary = list(tgt_vocab_set)

  layer2embs = defaultdict(list)
  vocab = []
  all_layers = {}
  ukn_encoded = tokenizer.encode(tokenizer.UNK_TOKEN, add_special_token=add_special_token, pad_to_max_length=False)[0]
  for entry in tqdm.tqdm(vocabulary, total=len(vocabulary)):
    sentences_masks = tokenizer.encode(entry, add_special_token=add_special_token, pad_to_max_length=False)
    if sentences_masks[0].numpy().tolist() != ukn_encoded.numpy().tolist():
      all_layers = encoding_model(sentences_masks=sentences_masks,
                                  sent_aggr=Modes.AVG,
                                  wp_aggr=Modes.AVG,
                                  is_single_instance=True)[0]
      for i, embedding_layer_i in enumerate(all_layers):
        layer2embs[i].append(embedding_layer_i)
      vocab.append(entry + "\n")

  assert len(all_layers) > 0
  for layer in range(len(all_layers)):
    target_file = basedir + "%s_layer_%s.npy" % (lang, str(layer))
    embeddings = np.array(layer2embs[layer], dtype=np.float)
    assert embeddings.shape[0] == len(vocab)
    with open(target_file, "wb") as f:
      np.save(f, embeddings)
  with open(basedir + "%s.vocab" % lang, "w") as f:
    f.writelines(vocab)


def main():
  MODELS = load_unspecialized_encoders()
  mbert = 0
  xlm = 1
  for model_name, Model, Tokenizer in [MODELS[mbert]]:
    for add_special_token in [False, True]:
      model = ModelWrapper(Model())
      tokenizer = Tokenizer()

      basedir = "/path/to/experiments/EncEval/ISO/"
      basedir += "model=%s/add_special_tokens=%s/" % (model_name, str(add_special_token))

      for lang in ["en"]: # , "it", "fi", "ru", "en"]:
        tmp_dir = basedir + "%s/" % lang
        os.makedirs(tmp_dir, exist_ok=True)
        print("\nrunning %s (%s)" % (lang, tmp_dir))
        run(tmp_dir, lang, model, tokenizer, add_special_token)
    print("All done for %s!" % model_name)

if __name__ == '__main__':
  main()

