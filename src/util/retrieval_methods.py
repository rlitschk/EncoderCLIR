from enum import Enum

class RetrievalMethod(Enum):
  INNER_PRODUCT = 0
  COSINE = 1
  WMD = 2 # Word Movers Distance