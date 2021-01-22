import os
import pickle
import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from bert import bert_tokenization
# from bert import tokenization
import src.config as c
os.environ["TFHUB_CACHE_DIR"] = c.TF_HUB_DIR

from src.experiment.evaluate import layer_wise_evaluation
from src.util.helper import chunk

_labse_cache = None
def laBSE_blackbox_experiment(query_lang, doc_lang, experiment_data, encode_fn, directory="", batch_size=4, **kwargs):
    global _labse_cache
    print("running labse blackbox experiment")

    assert kwargs["to_lower"] is True, "LaBSE model is uncased"
    if _labse_cache is None:
      max_seq_length =  kwargs["maxlen"]
      _labse_model, labse_layer = _get_laBSE_model(max_seq_length=max_seq_length)
      vocab_file = labse_layer.resolved_object.vocab_file.asset_path.numpy()
      do_lower_case = labse_layer.resolved_object.do_lower_case.numpy()
      # tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
      tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
      # tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
      _labse_cache = (_labse_model, labse_layer, tokenizer, max_seq_length)

    labse_model, labse_layer, tokenizer, max_seq_length = _labse_cache
    doc_ids, raw_documents, raw_queries, query_ids, relass = experiment_data

    raw_documents = [" ".join(line.lower().split()) for line in raw_documents]
    raw_queries = [" ".join(line.lower().split()) for line in raw_queries]

    def encode(input_text):
      input_ids, input_mask, segment_ids = _create_laBSE_input(
        input_text, tokenizer, max_seq_length)
      return labse_model([input_ids, input_mask, segment_ids])

    query_representations = []
    chunks = list(chunk(raw_queries, size=10))
    for batch in tqdm.tqdm(chunks, total=len(chunks)):
        current_representations = encode(batch).numpy()
        query_representations.extend(current_representations)
    query_representations = np.expand_dims(query_representations, 0)

    docs_cache = directory + "docs_%s_%s_%s.npy" % (doc_lang, str(len(raw_documents)), str(kwargs["maxlen"]))
    docids_cache = directory + "docids_%s_%s_%s.pkl" % (doc_lang, str(len(raw_documents)), str(kwargs["maxlen"]))
    os.makedirs(directory, exist_ok=True)
    if not (os.path.exists(docs_cache) and os.path.exists(docids_cache)):
      print("doc embeddings not found, creating new embeddings: %s" % docs_cache)
      document_representations = []
      chunks = list(chunk(raw_documents, size=10))
      for batch in tqdm.tqdm(chunks, total=len(chunks)):
          current_representations = encode(batch).numpy()
          document_representations.extend(current_representations)
      document_representations = np.expand_dims(document_representations, 0)
      with open(docs_cache, "wb") as f:
        np.save(f, document_representations)
      with open(docids_cache, "wb") as f:
        pickle.dump(doc_ids, f)
    else:
      print("loading cached doc representations: %s" % docs_cache)
      with open(docs_cache, "rb") as f:
        document_representations = np.load(f)
      with open(docids_cache, "rb") as f:
        doc_ids = pickle.load(f)

    lang_pair = query_lang + "-" + doc_lang
    result = layer_wise_evaluation(doc_ids, document_representations, query_ids, query_representations, relass,
                                   lang_pair=lang_pair,model_dir=directory)
    print("max-len: %s\tmap:%s" % (str(kwargs["maxlen"]), str(result[0])))
    return result


def _create_laBSE_input(input_strings, tokenizer, max_seq_length):
    input_ids_all, input_mask_all, segment_ids_all = [], [], []
    for input_string in input_strings:
        # Tokenize input.
        input_tokens = ["[CLS]"] + tokenizer.tokenize(input_string) + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        sequence_length = min(len(input_ids), max_seq_length)

        # Padding or truncation.
        if len(input_ids) >= max_seq_length:
            input_ids = input_ids[:max_seq_length]
        else:
            input_ids = input_ids + [0] * (max_seq_length - len(input_ids))

        input_mask = [1] * sequence_length + [0] * (max_seq_length - sequence_length)

        input_ids_all.append(input_ids)
        input_mask_all.append(input_mask)
        segment_ids_all.append([0] * max_seq_length)

    return np.array(input_ids_all), np.array(input_mask_all), np.array(segment_ids_all)


def _get_laBSE_model(max_seq_length, model_url="https://tfhub.dev/google/LaBSE/1"):
    labse_layer = hub.KerasLayer(model_url, trainable=True)

    # Define input.
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")

    # LaBSE layer.
    pooled_output,  _ = labse_layer([input_word_ids, input_mask, segment_ids])

    # The embedding is l2 normalized.
    pooled_output = tf.keras.layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1))(pooled_output)

    # Define model.
    return tf.keras.Model(
        inputs=[input_word_ids, input_mask, segment_ids],
        outputs=pooled_output), labse_layer