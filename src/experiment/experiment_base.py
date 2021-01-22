import os
import torch
import tqdm
import tempfile
import numpy as np

from src.experiment.evaluate import evaluate_map
from src.experiment.evaluate import layer_wise_evaluation
from src.experiment.index import retrieve
from src.util.timer import Timer as PPrintTimer
from src.util.embeddings import Embeddings
from src.model.text2vec import text2vec_idf_sum
from src.model.text2vec import create_text_representations

from typing import Callable
from typing import Dict
from typing import Tuple

timer = PPrintTimer()
MAP, PValue = float, float
Layer = int
EvaluationResult = Tuple[MAP, PValue]


def run_encoder_based_experiment(query_lang: str,
                                 doc_lang: str,
                                 experiment_data: tuple,
                                 encode_fn: Callable,
                                 directory: str="",
                                 batch_size: int=4,
                                 **kwargs) -> Dict[Layer, EvaluationResult]:
    """
    Constructs text representations according to specified encode_fn (usually implements pretrained supervised model) and
    performs retrieval and evaluation for provided queries and documents.
    :param query_lang: query language, e.g. en
    :param doc_lang: document language, e.g. de
    :param experiment_data: documents/queries with identifiers and relevance assessments
    :param encode_fn: either xlm or mbert
    :param directory: caching directory
    :param batch_size: encode n instances at a time, limited by GPU space.
    :return:
    """
    # unpacking values
    doc_ids, documents, query_ids, queries, relass = experiment_data
    num_docs = len(documents[0]) if len(documents) >= 3 else len(documents)
    doc_filename = doc_lang +  "_" + str(num_docs) + "_" + str(kwargs["maxlen"])  +  "doc.vectors.npy"
    num_queries = len(queries[0]) if len(queries) >= 3 else len(queries)
    query_filename = query_lang + "_" + str(num_queries) + "_" + str(kwargs["maxlen"])  + "query.vectors.npy"

    print("load/construct query representaions")
    query_representations = _load_representations(batch_size, query_lang, query_filename, queries, encode_fn,
                                                  cache_dir=directory, apply_procrustes=True)

    print("load/construct document representations")
    doc_representations = _load_representations(batch_size, doc_lang, doc_filename, documents, encode_fn,
                                                cache_dir=directory, apply_procrustes=False)
    print("Query- and Document-Embeddings created %s" % (timer.pprint_lap()))

    # best layer for xlm
    # doc_representations = doc_representations[11]
    # query_representations = query_representations[11]

    # best layer for mbert
    # doc_representations = doc_representations[9]
    # query_representations = query_representations[9]

    ap_results_dir = directory + "ap_results/"
    os.makedirs(ap_results_dir, exist_ok=True)
    kwargs["src_lang"] = query_lang
    kwargs["tgt_lang"] = doc_lang
    layer2result = layer_wise_evaluation(doc_ids, doc_representations, query_ids, query_representations, relass,
                                         savedir=ap_results_dir, lang_pair=query_lang + "-" + doc_lang,
                                         model_dir=directory, **kwargs)
    return layer2result


def _load_representations(batch_size: int,
                          language: str,
                          filename: str,
                          inpt: tuple,
                          encode_fn: Callable,
                          cache_dir: str="",
                          apply_procrustes: bool=False) -> np.ndarray:
    """
    Create/load (cached) sentence/document embeddings.

    :param batch_size: number of examples to embedd at once, e.g. 10 (limited by GPU space).
    :param language: @deprecated
    :param filename: filename for file in which representations are cached for later re-use.
    :param inpt: refer to encode function in run_ENC_exp.py
    :param encode_fn: instance of ModelWrapper
    :param cache_dir: folder where to cache representations.
    :param apply_procrustes: @deprecated
    :return:
    """
    if not cache_dir:
        cache_dir = tempfile.gettempdir()
        print("[WARNING] no cache directory specified, using %s" % cache_dir)

    filepath = cache_dir + filename
    if not os.path.exists(filepath):
        lines, masks, lengths, idfs, words_seqs = [], [], [], [], []

        array = []
        is_single_instance = True if batch_size == 1 else False
        for i, (doc, length, mask, idf_seq, word_seq, _) in enumerate(tqdm.tqdm(zip(*inpt), total=len(inpt[0]))):
            lines.append(doc)
            masks.append(mask)
            lengths.append(length)
            idfs.append(idf_seq)
            words_seqs.append(word_seq)

            if len(lines) > 0 and len(lines) % batch_size == 0:
                if is_single_instance:
                    raise NotImplementedError()

                result = encode_fn(sentences_masks=(lines, lengths, masks, idfs, words_seqs), language=language,
                                   apply_procrustes=apply_procrustes, is_single_instance=is_single_instance)
                if is_single_instance:
                    array.append(result)
                else:
                    array.extend(result)
                lines, masks, lengths, idfs, words_seqs = [], [], [], [], []

        if len(lines) > 0:
            array.extend(encode_fn((lines, lengths, masks, idfs, words_seqs), language=language,
                                   apply_procrustes=apply_procrustes))
        array = np.array(array, dtype=np.float32)

        if len(array.shape) == 3:
            array = array.transpose([1,0,2])

        os.makedirs(cache_dir, exist_ok=True)
        with open(filepath, "wb") as f:
            np.save(f, array)
    else:
        array = np.load(filepath)
    return array


def _pad_batch(lines):
    longest_seq = max([list(tmp.size())[0] for tmp in lines])
    newlines = []
    for line in lines:
        length = list(line.shape)[0]
        if length < longest_seq:
            pad = torch.tensor([(longest_seq - length) * [0]])
            newline = torch.cat((line, torch.squeeze(pad, 0)))
            newlines.append(newline)
        else:
            newlines.append(line)
    return newlines


def run_we_based_experiment(aggregation_fn: Callable,
                            query_lang: str,
                            doc_lang: str,
                            experiment_data: tuple,
                            initialized_embeddings: Embeddings,
                            processes: int=40,
                            to_lower: bool=False,
                            **kwargs) -> MAP:
    """
    Constructs word-embedding based text representations for queries and documents according to the specified aggregation
    method. From the text representations it retrieves for each query the documents and computes the evaluation metric.

    :param aggregation_fn: text2vec_idf_sum or text2vec_sum
    :param query_lang: e.g. EN
    :param doc_lang: e.g. DE
    :param experiment_data: document ids, documents, query ids, queries, relevance assessments
    :param initialized_embeddings: util.embeddings
    :param processes: number of processes (multiprocessing pool)
    :param to_lower: whether to lowercase text
    :return: mean average precision
    """
    # unpacking values
    qlang_short, qlang_long = query_lang
    dlang_short, dlang_long = doc_lang
    doc_ids, documents, query_ids, queries, relass = experiment_data
    embeddings = initialized_embeddings
    assert embeddings.emb_sizes[dlang_long] == embeddings.emb_sizes[qlang_long]

    doc_arry = create_text_representations(language=dlang_long, id_text=zip(doc_ids, documents),
                                           emb=embeddings, processes=processes, aggregation=aggregation_fn,
                                           idf_weighing=aggregation_fn == text2vec_idf_sum,
                                           emb_dim=embeddings.emb_sizes[dlang_long], to_lower=to_lower)
    query_arry = create_text_representations(language=qlang_long, id_text=zip(query_ids, queries),
                                             emb=embeddings, processes=processes, aggregation=aggregation_fn,
                                             idf_weighing=False, emb_dim=embeddings.emb_sizes[qlang_long],
                                             to_lower=to_lower)  # Queries are not idf-scaled
    print("Query- and Document-Embeddings created %s" % (timer.pprint_lap()))

    kwargs["src_lang"] = qlang_short
    kwargs["tgt_lang"] = dlang_short
    lang_pair = qlang_short+"-"+dlang_short
    experiment_name = kwargs.get("emb_space_method", "UKN") + "_" + lang_pair

    # Run retrieval model
    query2ranking = retrieve(doc_arry, doc_ids, query_arry, query_ids, dim=embeddings.emb_sizes[qlang_long])

    # Evaluate all rankings
    MAP, p_value = evaluate_map(query2ranking=query2ranking, relass=relass, experiment_name=experiment_name,
                                     lang_pair=lang_pair, **kwargs)
    return MAP
