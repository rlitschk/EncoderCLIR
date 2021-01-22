import os

from itertools import islice

def rm_digits(txt):
  return ''.join(filter(lambda x: not x.isdigit(), txt))


def chunk(it, size):
  it = iter(it)
  return iter(lambda: tuple(islice(it, size)), ())


def create_dir(directory):
  if not directory.endswith("/"):
    directory += "/"
  if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)