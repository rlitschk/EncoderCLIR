import unittest
import numpy as np
from src.experiment.index import retrieve


np.random.seed(123)

class TestRanker(unittest.TestCase):

  def test_single_query(self):
    emb_size = 10
    ndocs = 5

    query = np.ones(shape=[1, emb_size]).astype("float32")

    # documents ordered by decreasing relevance (cosine similarity)
    documents = np.ones(shape=[ndocs, emb_size]).astype("float32")
    documents = np.triu(documents)

    doc_ids = ["doc_" + str(_id) for _id in range(ndocs)]
    query_ids = ["query_" + str(_id) for _id in range(1)]

    correct_ranking = doc_ids
    query2ranking = retrieve(doc_arry=documents, doc_ids=doc_ids, query_arry=query,
                             dim=emb_size, query_ids=query_ids)
    ranking = query2ranking[query_ids[0]]
    for i, docid in enumerate(ranking):
      self.assertEqual(docid, correct_ranking[i])

    query = np.zeros(shape=[1, emb_size]).astype("float32")
    query[0][0] = 1 # one hot query
    query2ranking = retrieve(doc_arry=documents, doc_ids=doc_ids, query_arry=query,
                             dim=emb_size, query_ids=query_ids)
    ranking = query2ranking[query_ids[0]]
    # first document == relevant (cos > 0), all other documents have equal similarity (cos = 0)
    self.assertEqual(correct_ranking[0], ranking[0])

  def test_multiple_queries(self):
    n_queries = 20
    n_docs = 100
    emb_size = 50

    query_ids = [str(i) for i in range(n_queries)]
    queries = np.random.normal(size=[n_queries, emb_size]).astype("float32")

    doc_ids = [str(i) for i in range(n_docs)]
    documents = np.random.normal(size=[n_docs, emb_size]).astype("float32")

    query2ranking = retrieve(doc_arry=documents, query_arry=queries, doc_ids=doc_ids, query_ids=query_ids, dim=emb_size)

    for qid, query in zip(query_ids, queries):
      # manual ranking
      similarities = np.matmul(query, documents.T)
      ref_ranking = [doc_ids[i] for i in np.argsort(-similarities)]
      ranking = query2ranking[qid]

      self.assertEqual(len(ranking), len(ref_ranking))
      for docid, ref_docid in zip(ranking, ref_ranking):
        self.assertEqual(docid, ref_docid)
