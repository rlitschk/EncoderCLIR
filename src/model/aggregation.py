import numpy as np
import torch
from enum import Enum

class Modes(Enum):
  FIRST = 1
  ALL = 2
  MAX = 3
  AVG = 4
  SUM = 5

  def __str__(self):
    return self.name


def aggregate_torch(embeddings, mode, length=None):
  """
  PyTorch embedding aggregation
  :param embeddings: sequence of word embeddings
  :param mode: e.g. Modes.AVG
  :param length: sequence length
  :return:
  """
  assert len(embeddings) > 0
  if len(embeddings) == 1:
    return embeddings[0]

  if mode == Modes.SUM:
    sequence_summary = torch.sum(embeddings, dim=0)
  elif mode == Modes.AVG:
    if length is None:
      sequence_summary = torch.mean(embeddings, dim=0)
    else:
      sequence_summary = torch.sum(embeddings, dim=0) / length
  elif mode == Modes.FIRST:
    sequence_summary = embeddings[0]
  elif mode == Modes.MAX:
    sequence_summary, _ = torch.max(embeddings, dim=0)
  else:
    assert mode == Modes.ALL
    sequence_summary = embeddings

  # assert torch.count_nonzero(sequence_summary) > 0
  return sequence_summary



def aggregate(embeddings, mode, length=None):
  """
  Numpy embedding aggregation
  :param embeddings: sequence of word embeddings
  :param mode: e.g. Modes.AVG
  :param length: sequence length
  :return:
  """
  assert len(embeddings) > 0
  if len(embeddings) == 1:
    return embeddings[0]

  if mode == Modes.AVG:
    assert length is not None
    sequence_summary = np.sum(embeddings, axis=0) / length
  elif mode == Modes.FIRST:
    sequence_summary = embeddings[0]
  elif mode == Modes.MAX:
    sequence_summary = np.max(embeddings, axis=0)
  else:
    assert mode == Modes.ALL
    sequence_summary = embeddings

  assert np.count_nonzero(sequence_summary) > 0
  return sequence_summary
