from itertools import compress

from typing import List
from typing import Dict
from typing import Tuple

import faiss
import numpy as np

from src.util.timer import Timer as PPrintTimer


MAP, PValue = float, float
Layer = int
Query_ID, Document_ID = str, str
EvaluationResult = Tuple[MAP, PValue]


def create_index(vectors: np.ndarray, dim: int=300) -> Tuple[faiss.IndexIVFFlat, faiss.IndexFlatIP]:
    """
    Returns a faiss index built and trained on :param vectors for inner product serach.

    If you modify this function, be careful with this:
    https://github.com/facebookresearch/faiss/issues/45

    Also: "There is no guarantee of ordering when distances are the same."
    https://github.com/facebookresearch/faiss/issues/1201

    :param vectors: document vectors
    :param dim: embedding size
    :return: faiss index (index, quantizer
    """
    nlist = nprobe = 5
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = nprobe
    return index, quantizer


def retrieve(doc_arry: np.ndarray,
             doc_ids: List[Document_ID],
             query_arry: np.ndarray,
             query_ids: List[Query_ID],
             dim: int=300,
             verbose: bool=True,
             timer: PPrintTimer=None) -> Dict[Query_ID, List[Document_ID]]:
    """
    Constructs index on document embeddings and run queries on it. Document and query embeddings need to have the same
    size. Expected shape: n_docs x hidden_size / n_queries x hidden_size. Returns ranking for each query. Documents
    with zero vectors (usually empty documents) are removed.

    :param doc_arry: Document embeddings
    :param doc_ids: Document IDs
    :param query_arry: Query embeddings
    :param query_ids: Query IDs
    :param dim: Embedding size
    :param verbose: log when retrieval done
    :param timer: timer
    :return: mapping from query id to ranking of document ids
    """
    rankings, doc_ids = _retrieve_IP(doc_arry=doc_arry, doc_ids=doc_ids, query_arry=query_arry,
                                     dim=dim, verbose=verbose, timer=timer)
    rankings_with_doc_ids = np.array(doc_ids)[rankings]
    query2ranking = {query_ids[i]: rankings_with_doc_ids[i] for i in range(len(query_ids))}
    return query2ranking


def _retrieve_IP(doc_arry: np.ndarray,
                 doc_ids: List[Document_ID],
                 query_arry: np.ndarray,
                 dim: int=300,
                 topk: int=-1,
                 verbose: bool=True,
                 timer: PPrintTimer=None) -> Tuple[np.ndarray, List[Document_ID]]:
    """
    Retrieve Inner Product: Retrieves results from doc_arry for all queries in querry_arry using inner product. Length-normalizing
    vectors in doc_arry and query_arry leads to cosine similarity retrieval. We keep only documents for which
    we have a non-zero text embedding, i.e. for which at least one word embedding could exists (filters out empty documents)
    Important: id lists and embedding matrices must be aligned.

    :param doc_arry: numpy array containing document vectors
    :param doc_ids: numerical identifiers
    :param query_arry: numpy array containing query vectors
    :param dim: embedding size
    :param timer: timer
    :param verbose: log when retrieval done
    :return: list of rankings (one for each query) and filtered document ids (keep those with non-empty embs)
    """
    if not timer:
        timer = PPrintTimer()

    doc_non_zero = np.array([np.count_nonzero(doc) > 0 for doc in doc_arry]) # np.all(doc_arry != 0, axis=1)
    doc_arry = doc_arry[doc_non_zero]
    doc_ids = list(compress(doc_ids, doc_non_zero))
    index, quantizer = create_index(doc_arry, dim)
    if topk == -1:
        topk = len(doc_arry)
    distances, rankings = index.search(query_arry, topk)
    if verbose:
        print("Retrieval done %s" % (timer.pprint_lap()))
    return rankings, doc_ids
