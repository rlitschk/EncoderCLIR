import unittest
import numpy as np

from src.experiment.evaluate import evaluate_map


class TestEval(unittest.TestCase):

  def test_multiple_queries(self):
    query2ranking = {
      "x": ["a", "b", "c", "d"],
      "y": ["d", "b", "c", "a"],
      "z": ["a", "c", "b", "d"] }
    relass = { "x": ["c"],  "y": ["a", "b"], "z": ["d"] }

    # assert correctness of MAP
    MAP = evaluate_map(query2ranking=query2ranking, relass=relass)[0]
    AP_x = 1 / 3
    AP_y = 1 / 2 * (2 / 4 + 1 / 2)
    AP_z = 1 / 4
    manual_MAP = 1 / 3 * (AP_x + AP_y + AP_z)
    self.assertAlmostEqual(manual_MAP, MAP)

    # assert indifference of MAP towards additional relevance assessments
    relass["p"] = ["q"]
    eq_MAP = evaluate_map(query2ranking=query2ranking, relass=relass)[0]
    self.assertAlmostEqual(MAP, eq_MAP)

  def test_different_rankings(self):
    query2ranking = { "x": ["a", "b", "c"] }
    relass = {"x": ["b"]}
    MAP = evaluate_map(query2ranking, relass)[0]

    # test MAP decrease for worse ranking
    query2ranking = {"x": ["a", "c", "b"]}
    worse_MAP = evaluate_map(query2ranking, relass)[0]
    self.assertLess(worse_MAP, MAP)

    # test MAP improvement for improved ranking
    query2ranking = {"x": ["b", "c", "a"]}
    improved_MAP = evaluate_map(query2ranking, relass)[0]
    self.assertGreater(improved_MAP, MAP)

  def test_relevant_doc_missing_in_ranking(self):
    query2ranking = {"x": ["a", "b", "c"], "y": ["b", "a", "c"]}

    # single query with missing relevant doc
    relass = {"y": ["missing"]}
    MAP = evaluate_map(query2ranking, relass)[0]
    self.assertTrue(np.isnan(MAP))

    # multipe queries with one ranking missing relevant doc
    relass = {"x": ["b"], "y": ["missing"]}
    MAP_x = evaluate_map(query2ranking, relass)[0]
    self.assertEqual(MAP_x, 1/2)

    # MAP is not a function of number of missing docs
    relass = {"x": ["b"]}
    MAP_ref = evaluate_map(query2ranking, relass)[0]
    self.assertEqual(MAP_x, MAP_ref)
