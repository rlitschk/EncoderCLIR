import os
import pickle
import numpy as np
import torch
import tqdm

from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

from src.config import distilmodel2path
from src.dataloaders.paths_corpus import get_lang2pair
from src.experiment.evaluate import layer_wise_evaluation
from src.experiment.experiment_base import timer
from src.util.helper import chunk
import src.config as c


os.environ["TORCH_HOME"] = c.CKPT_DIR + "torch/"
_cache = None

def distil_blackbox_experiment(query_lang, doc_lang, experiment_data, encode_fn, directory="", batch_size=1000, **kwargs):
    """
    performances (fi-ru), no idf-scaling:
    "xlm-r-100langs-bert-base-nli-stsb-mean-tokens": 0.14281965565401758
    "distiluse-base-multilingual-cased": 0.10989724755148565
    "xlm-r-100langs-bert-base-nli-mean-tokens": 0.06291587497944251
    "distilbert-multilingual-nli-stsb-quora-ranking":  0.2599846340011951
    """
    global  _cache
    print("running DISTIL blackbox %s experiment" % encode_fn)

    doc_ids, documents, query_ids, queries, relass = experiment_data
    documents = list(documents)

    rm_stopwords = False

    doc_ids, documents = filter_empty_lines(doc_ids, doc_lang, documents, rm_stopwords=rm_stopwords)
    query_ids, queries = filter_empty_lines(query_ids, query_lang, queries, rm_stopwords=rm_stopwords)

    if not _cache:
        _path = distilmodel2path[encode_fn]
        if os.path.exists(_path):
            print("loading from checkpoint: %s" % _path)
            embedder = SentenceTransformer(_path)
        else:
            print("checkpoint not found in %s, downloading model" % _path)
            embedder = SentenceTransformer(encode_fn)
        embedder.max_seq_length = kwargs['maxlen']
        embedder.eval()
        embedder = embedder.cuda()
        _cache = embedder
    embedder = _cache

    query_representations = embedder.encode(queries, show_progress_bar=True, convert_to_tensor=False,
                                             convert_to_numpy=True, is_pretokenized=False,
                                             output_value="sentence_embedding")
    print("query-Embeddings created %s" % (timer.pprint_lap()))

    docs_cache = directory + "docs_%s_%s_%s.npy" % (doc_lang, str(len(documents)), str(kwargs['maxlen']))
    docids_cache = directory + "docids_%s_%s_%s.pkl" % (doc_lang, str(len(documents)), str(kwargs['maxlen']))
    os.makedirs(directory, exist_ok=True)
    if not (os.path.exists(docs_cache) and os.path.exists(docids_cache)):
        print("doc embeddings not found, creating new embeddings: %s" % docs_cache)

        batches = list(chunk(documents, size=batch_size))
        doc_representations = []
        for document_batch in tqdm.tqdm(batches, total=len(batches)):
            current_document_embeddings = embedder.encode(document_batch, show_progress_bar=False, output_value="sentence_embedding",
                                                       is_pretokenized=False, convert_to_numpy=True, convert_to_tensor=False)
            doc_representations.extend(current_document_embeddings)
        doc_representations = np.array(doc_representations)

        with open(docs_cache, "wb") as f:
            np.save(f, doc_representations)
        with open(docids_cache, "wb") as f:
            pickle.dump(doc_ids, f)
    else:
        print("loading cached doc representations: %s" % docs_cache)
        with open(docs_cache, "rb") as f:
            doc_representations = np.load(f)
        with open(docids_cache, "rb") as f:
            doc_ids = pickle.load(f)

    lang_pair = query_lang + "-" + doc_lang
    layer2result = layer_wise_evaluation(doc_ids, doc_representations, query_ids, query_representations, relass,
                                   lang_pair=lang_pair, model_dir=directory)
    print("max-len: %s\tmap:%s" % (str(kwargs["maxlen"]), str(layer2result[0])))
    return layer2result


def filter_empty_lines(ids, language, lines, rm_stopwords=False):
    tmp_ids = []
    tmp_docs = []
    if rm_stopwords:
        stpwords = set(stopwords.words(get_lang2pair(language)[1]))
        print("stopword list: %s" % str(len(stpwords)))
    else:
        stpwords = set()

    for _id, doc in zip(ids, lines):
        if doc.strip() != "":
            tmp_ids.append(_id)
            if rm_stopwords:
                tmp_docs.append(" ".join([term for term in doc.split() if term not in stpwords]))
            else:
                tmp_docs.append(" ".join(doc.split()))
    lines = tmp_docs
    ids = tmp_ids
    return ids, lines


def _aggregate_token_embeddings(aggregator, wp_aggr, sent_aggr, lines, language, all_token_embeddings):
    print("aggregate %s token embeddings" % language)
    tmp = []
    it = zip(all_token_embeddings, *lines)
    for token_embeddings, _, length, mask, idfs, words, _ in tqdm.tqdm(it, total=len(all_token_embeddings)):
        embedded_sequence = torch.unsqueeze(token_embeddings, 0)
        emb = aggregator.generate_sentence_embeddings(embedded_sequence, masks=[mask], idfs=[idfs],
                                                      word_seqs=[words],
                                                      lengths=[torch.min(torch.tensor([length, 131]))],
                                                      language=language, word_aggr=sent_aggr,
                                                      wp_aggr=wp_aggr)
        tmp.append(emb.detach().cpu().numpy())
    ref_query_representations = np.concatenate(tmp, axis=0)
    ref_query_representations = np.expand_dims(ref_query_representations, 0)  # add layer dimension
    return ref_query_representations
