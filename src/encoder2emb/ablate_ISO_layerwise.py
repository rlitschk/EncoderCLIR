import pickle
import numpy as np
import os
from src.encoder2emb.ablate_AOC_ctxfreqwise import save
from src.encoder2emb.ablate_AOC_ctxfreqwise import map_proc
from src.run_ENC_exp import print_store_results
from src.run_CLWE_exp import run

lang_pairs = [("en", "fi"), ("en", "it"), ("en", "ru"), ("en", "de"),
              ("de", "fi"), ("de", "it"), ("de", "ru"),
              ("fi", "it"), ("fi", "ru")]
# lang_pairs = [("de", "it")]
dataset = "clef"
add_special_tokens=True
print("dataset=%s" % dataset)
results = []
mode_layers = [("custom_xlm-mlm-100-1280", 17), # 0
               ("hf_bert-base-multilingual-cased", 13), # 1
               ("hf_bert-base-multilingual-uncased", 13), # 2   <----- 2
               ("hf_xlm-mlm-100-1280", 17), # 3                 <----- 3
               ("hf_xlm-roberta-base", 13), # 4
               ("hf_xlm-roberta-large", 25)] # 5

xlm=3
mbert=2
selected_model = [mode_layers[mbert]]
layer2result = {}
assert len(selected_model) == 1
for model, num_layers in selected_model:
  path = "/work/usr/experiments/EncEval/ISO/model=%s/add_special_tokens=%s/" % (model, str(add_special_tokens))
  print("base path: %s" % path)

  for layer in range(0, 1):
    layer_results = []
    print("evaluating layer %s" % str(layer))
    for l1, l2 in lang_pairs:
      print("%s->%s" % (l1, l2))

      if (l1 == "ru" or l2 == "ru") and dataset == "europarl":
        continue

      proc_base_path = path + "mapped_proc/%s-%s_%s/" % (l1, l2, str(layer))
      os.makedirs(proc_base_path, exist_ok=True)

      proc_l1_emb_path = proc_base_path + "%s-%s.%s.vectors" % (l1, l2, l1)
      proc_l2_emb_path = proc_base_path + "%s-%s.%s.vectors" % (l1, l2, l2)
      proc_l1_vocab_path = proc_l1_emb_path.replace("vectors", "vocab")
      proc_l2_vocab_path = proc_l2_emb_path.replace("vectors", "vocab")
      path_proj = proc_base_path + l1 + "-" + l2 + ".proj"

      to_lower = True if model == "hf_bert-base-multilingual-uncased" else False

      all_paths_exists = True
      for tmp_path in [proc_l1_emb_path, proc_l2_emb_path, proc_l1_vocab_path, proc_l2_vocab_path]:
        if not os.path.exists(tmp_path):
          all_paths_exists = False
          print("%s not found, running procrustes now" % tmp_path)
          break

      if not all_paths_exists:
        src_emb_path = path + "%s/%s_layer_%s.npy" % (l1, l1, str(layer))
        tgt_emb_path = path + "%s/%s_layer_%s.npy" % (l2, l2, str(layer))
        with open(src_emb_path, "rb") as f:
          src_embeddings = np.load(f)
        with open(tgt_emb_path, "rb") as f:
          tgt_embeddings = np.load(f)
        src_vocab_path = path + "%s/%s_vocab.pkl" % (l1, l1)
        with open(src_vocab_path, "rb")  as f:
          src_vocab = pickle.load(f)
        tgt_vocab_path = path + "%s/%s_vocab.pkl" % (l2, l2)
        with open(tgt_vocab_path, "rb") as f:
          tgt_vocab = pickle.load(f)

        src_term2emb = {term: src_embeddings[i] for term, i in src_vocab.items()}
        tgt_term2emb = {term: tgt_embeddings[i] for term, i in tgt_vocab.items()}

        proj_mat, src_emb, src_vocab, tgt_emb, tgt_vocab = map_proc(l1, l2, src_term2emb, tgt_term2emb, to_lower)
        print(proc_l1_emb_path)
        with open(proc_l1_emb_path, "wb") as f:
          np.save(f, src_emb)
        print(proc_l1_vocab_path)
        save(proc_l1_vocab_path, src_vocab)

        print(proc_l2_emb_path)
        with open(proc_l2_emb_path, "wb") as f:
          np.save(f, tgt_emb)
        print(proc_l2_vocab_path)
        save(proc_l2_vocab_path, tgt_vocab)

        print(path_proj)
        with open(path_proj, "wb") as f:
          np.save(f, proj_mat)

      map_score = run(query_lang=l1,
                      path_query_embeddings=proc_l1_emb_path,
                      path_query_vocab=proc_l1_vocab_path,
                      doc_lang=l2,
                      path_doc_embeddings=proc_l2_emb_path,
                      path_doc_vocab=proc_l2_vocab_path,
                      retrieval_method="IDF-SUM", dataset=dataset,
                      to_lower=to_lower)
      layer2result[layer] = map_score
      layer_results.append(map_score)
    results.append(layer_results)


num_models = 12
num_langpairs = 9 if dataset == "clef" else 6
bonferroni = num_models * num_langpairs
print_store_results(path, model, layer2result, bonferroni=bonferroni)

print("done!")
# print("context-count = %s" % str(context_size))
for layer_results in results:
  print("\t".join([str(result) for result in layer_results]))