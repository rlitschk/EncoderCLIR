import os
import numpy as np
from src import config

from src.experiment.index import retrieve
from src.util.timer import Timer as PPrintTimer
from scipy.stats import ttest_ind

from typing import Dict
from typing import List
from typing import Tuple


MAP, PValue = float, float
Layer = int
Query_ID, Document_ID = str, str
EvaluationResult = Tuple[MAP, PValue]


def layer_wise_evaluation(doc_ids: List[Document_ID],
                          doc_representations: np.ndarray,
                          query_ids: List[Query_ID],
                          query_representations: np.ndarray,
                          relass: Dict[Query_ID, List],
                          chunk_size = 100,
                          timer: PPrintTimer=None,
                          l2normalization: bool=True,
                          **kwargs) -> Dict[Layer, EvaluationResult]:
    """
    Evaluates each layer of a given encoder, e.g. 12-layers for mbert. Performs chunk-wise evaluation to avoid space
    issues.

    :param doc_ids: document ids
    :param doc_representations: expected shape: [num_layers x num_exampes x hidden-size] or [num_exampes x hidden-size]
    :param query_ids: query ids
    :param query_representations: expected shape: [num_layers x num_exampes x hidden-size] or [num_exampes x hidden-size]
    :param relass: relevance assessments {qid: [doc_ids]}
    :param chunk_size: number of queries run at the same time (retrieval)
    :param timer: instance of PPrintTimer
    :param l2normalization: whether to length normalize embeddings
    :param kwargs: significance tests require 'src_lang' (e.g. 'en'), 'tgt_lang' (e.g. 'de'), 'dataset' (e.g. 'europarl')
    :return: {layer: (MAP, p-value)}
    """
    if not timer:
        timer = PPrintTimer()
    print("running chunk-wise (chunk-size=%s) evaluation %s" % (str(chunk_size), timer.pprint_lap()))

    # add layer dimension if necessary
    if len(doc_representations.shape) == 2:
        doc_representations = np.expand_dims(doc_representations, 0)
    if len(query_representations.shape) == 2:
        query_representations = np.expand_dims(query_representations, 0)

    if l2normalization:
        query_representations = np.array([_normalize_vecs(layer_matrix) for layer_matrix in query_representations], dtype=np.float32)
        print("Queries length normalized %s" % (timer.pprint_lap()))

        doc_representations = np.array([_normalize_vecs(layer_matrix) for layer_matrix in doc_representations], dtype=np.float32)
        print("Documents length normalized %s" % (timer.pprint_lap()))

    # Expected input: num_layers x num_examples x hidden-size
    num_layers = query_representations.shape[0]
    num_queries = query_representations.shape[1]
    embedding_size = query_representations.shape[2]

    layer2result = {}
    # Run layer-wise evaluation
    for i in range(num_layers):
        print("Evaluating layer %s" % str(i))
        current_query_representations = query_representations[i]
        current_doc_representaions = doc_representations[i]

        # run queries in chunks, i.e. not all queries at once
        query2ranking = {}
        for chunk_i, start_range in enumerate(range(0, num_queries, chunk_size)):
            # get offset
            end_range = start_range + chunk_size

            # fetch current batch of queries
            subset_query_representations = current_query_representations[start_range:end_range]
            subset_qids = query_ids[start_range:end_range]

            # run retrieval
            tmp_query2ranking = retrieve(doc_arry=current_doc_representaions,
                                         doc_ids=doc_ids,
                                         query_arry=subset_query_representations,
                                         query_ids=subset_qids,
                                         dim=embedding_size)
            query2ranking.update(tmp_query2ranking)
            print("%s" % str(chunk_i), end="\t")

        # combine ranking of all queries evaluated in chunks
        # rankings = list(chain(*tmp_rankings))
        if kwargs.get('split_documents', False):
            tmp = {}
            for qid, ranking in query2ranking:
                cleaned_ranking = []
                considered_docs = set()
                for doc in ranking:
                    doc_id = doc.split("_")[0]
                    if doc_id not in considered_docs:
                        cleaned_ranking.append(doc_id)
                        considered_docs.add(doc_id)
                tmp[qid] = cleaned_ranking
            query2ranking = tmp

        evaluation_result = evaluate_map(query2ranking=query2ranking, relass=relass,  layer=i, **kwargs)
        layer2result[i] = evaluation_result
    print({k: round(v[0], 4) for k, v in layer2result.items()})
    return layer2result


def evaluate_map(query2ranking: Dict[Query_ID, List[Document_ID]],
                 relass: Dict[Query_ID, List[Document_ID]],
                 **kwargs) -> EvaluationResult:
    """
    Evaluates results for queries in terms of Mean Average Precision (MAP). Evaluation gold standard is
    loaded from the relevance assessments.

    :param query2ranking: (actual) ranking for each query
    :param relass: gold standard (expected) ranking for each query
    :return: tuple(MAP, p-value)
    """
    # collect AP values for MAP
    average_precision_values = []

    # collect all precision values for significance test
    all_precisions = []

    for query_id, ranking in query2ranking.items():
        if query_id in relass:  # len(relevant_docs) > 0:
            relevant_docs = relass[query_id]

            # get ranking for j'th query
            is_relevant = [document in relevant_docs for document in ranking]
            ranks_of_relevant_docs = np.where(is_relevant)[0].tolist()

            precisions = []
            # +1 because of mismatch betw. one based rank and zero based indexing
            for k, rank in enumerate(ranks_of_relevant_docs, 1):
                precision_at_k = k / (rank + 1)
                precisions.append(precision_at_k)
            all_precisions.extend(precisions)

            if len(precisions) == 0:
                print("Warning: query %s without relevant documents in corpus: %s (skipped)" % (query_id, relevant_docs))
            else:
                ap = np.mean(precisions)
                average_precision_values.append(ap)

    mean_average_precision = np.mean(np.array(average_precision_values))
    mean_average_precision = float(mean_average_precision)
    pvalue = -1.0

    # Significance test against reference model (proc-B)
    if 'src_lang' in kwargs and 'tgt_lang' in kwargs and 'dataset' in kwargs:
        PROJECT_ROOT = os.path.abspath(os.path.dirname(config.__file__))
        filename = 'procb_%s-%s.txt' % (kwargs['src_lang'], kwargs['tgt_lang'])
        file = os.path.join(PROJECT_ROOT, 'data', 'ttest-references','%s-precision-values' % kwargs['dataset'], filename)
        if os.path.exists(file):
            with open(file, "r") as f:
                reference_precision_values = [float(line.strip()) for line in f.readlines()]
            pvalue = ttest_ind(reference_precision_values, all_precisions)[1]

    return mean_average_precision, pvalue


def _normalize_vecs(arry: np.ndarray) -> np.ndarray:
    """
    Apply l2-normalization alongside the last dimension.

    :param arry: expected shape: [num_embeddings x hidden_size]
    :return: l2-normalized embeddings
    """
    normalized_arry = []
    for elem in arry:
        elem_norm = np.linalg.norm(elem)
        if elem_norm == 0:
            elem_norm = 1
        normalized_arry.append(elem / elem_norm)
    normalized_arry = np.array(normalized_arry)
    return normalized_arry
