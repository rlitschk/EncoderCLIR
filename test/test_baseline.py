import numpy as np
import unittest
from src.experiment.baselines import score_unigram_lm
from collections import Counter


class TestBaseline(unittest.TestCase):
  LAMBDA = 0.3

  @staticmethod
  def get_test_data():
    p_qt_d1 = [0.3, 0.5, 0.1]
    p_qt_d2 = [0.5, 0.8, 0.2]
    p_qt_C = [0.8, 0.1, 0.4]
    return p_qt_d1, p_qt_d2, p_qt_C

  @staticmethod
  def get_test_LMs(num_docs):
    collection_vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    docs = []
    for _ in range(num_docs):
      random_doclen = np.random.randint(0, 20)
      docs.append([collection_vocab[np.random.randint(0, len(collection_vocab))] for _ in range(random_doclen)])

    doc_freq_distributions = [Counter(doc) for doc in docs]
    collection_freq_distr = doc_freq_distributions[0]
    for lm in doc_freq_distributions:
      collection_freq_distr.update(lm)

    _total = sum(collection_freq_distr.values())
    collection_LM = {term: freq/_total for term, freq in collection_freq_distr.items()}
    doc_LMs = []
    for doc_freq_dist in doc_freq_distributions:
      _total = sum(doc_freq_dist.values())
      doc_LMs.append({k: v/_total for k,v in doc_freq_dist.items()})
    return collection_LM, doc_LMs, collection_freq_distr, doc_freq_distributions

  @staticmethod
  def run_unigramlm_linspace(p_qt_d, p_qt_C, lmbda):
    prob = 1
    for i in range(3):
      x = lmbda * p_qt_d[i] + (1 - lmbda) * p_qt_C[i]
      prob = prob * x
    return prob

  @staticmethod
  def run_unigramlm_logspace(p_qt_d, p_qt_C, lmbda):
    log_prob = 0
    for i in range(len(p_qt_d)):
      x = np.log(lmbda * p_qt_d[i] + (1 - lmbda) * p_qt_C[i])
      log_prob += x
    return log_prob

  def test_linspace_logspace_equal(self):
    p_qt_d1, p_qt_d2, p_qt_C = TestBaseline.get_test_data()
    a = self.run_unigramlm_linspace(p_qt_d=p_qt_d1, p_qt_C=p_qt_C, lmbda=0.3)
    b = self.run_unigramlm_logspace(p_qt_d=p_qt_d1, p_qt_C=p_qt_C, lmbda=0.3)
    b = np.exp(b)
    self.assertAlmostEqual(a, b)

  def test_unigram_baseline_impl(self):
    collection_LM, doc_LMs, collection_freq_distr, doc_freq_distributions = TestBaseline.get_test_LMs(5)
    queries = [['a', 'b', 'c', 'a'], ['x', 'y'], ['a', 'x']]
    mu = 100

    doc_lengths = {sum(df.values()) for df in doc_freq_distributions}
    doclen2lambda = {dl: dl/(dl+mu) for dl in doc_lengths}
    lambdas = np.array([doclen2lambda[sum(doc_freq.values())] for doc_freq in doc_freq_distributions])
    lambdas = np.expand_dims(lambdas, 1)


    for query in queries:
      p_qt_C = [collection_LM.get(term,0) for term in query]
      ref_scores = []
      for doc_LM, doc_freq_distr in zip(doc_LMs, doc_freq_distributions):
        p_qt_d = [doc_LM.get(term ,0) for term in query]
        doc_len = sum(doc_freq_distr.values())
        _lambda = doclen2lambda[doc_len]
        lm_score = self.run_unigramlm_logspace(p_qt_C=p_qt_C, p_qt_d=p_qt_d, lmbda=_lambda)
        ref_scores.append(lm_score)
      ref_scores = np.array(ref_scores)
      lm_scores = score_unigram_lm(collection_LM=collection_LM, lambdas=lambdas, document_LMs=doc_LMs, query=query)
      np.testing.assert_equal(ref_scores, lm_scores)

  def test_vectorized_LM(self):
    p_qt_d1, p_qt_d2, p_qt_C = TestBaseline.get_test_data()

    a = np.array(p_qt_d1)
    b = np.array(p_qt_d2)
    ab = np.array([a, b])

    z = np.array(p_qt_C)

    lambdas = np.array([0.3, 0.5])
    lambdas = np.expand_dims(lambdas, axis=1)

    log_probs = np.log(lambdas * ab + (1 - lambdas) * z).sum(axis=1)
    log_prob1 = self.run_unigramlm_logspace(p_qt_d=p_qt_d1, p_qt_C=p_qt_C, lmbda=lambdas[0])
    log_prob2 = self.run_unigramlm_logspace(p_qt_d=p_qt_d2, p_qt_C=p_qt_C, lmbda=lambdas[1])

    self.assertEqual(log_probs[0], log_prob1)
    self.assertEqual(log_probs[1], log_prob2)
