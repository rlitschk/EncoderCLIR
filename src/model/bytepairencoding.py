import subprocess


fastBPE_path = "/path/to/projects/XLM/tools/fastBPE/fast"
XLM_BPE_CODES = "/path/to/data/pretrained_xlm/mlm_100_1280/codes_xnli_100.txt"
XLM_BPE_VOCAB = "/path/to/data/pretrained_xlm/mlm_100_1280/vocab_xnli_100.txt"


class InteractiveBPE:

  def __init__(self, codes=None, vocab=None):
    if codes is None:
      print("WARNING: using default BPE codes (XLM mlm 100 model)")
      self.codes = XLM_BPE_CODES
    else:
      self.codes = codes
    if vocab is None:
      print("WARNING: using default BPE vocab (XLM mlm 100 model)")
      self.vocab = XLM_BPE_VOCAB
    else:
      self.vocab = vocab
    self.proc = None

  def open(self):
    self.proc = subprocess.Popen([fastBPE_path, "applybpe_stream", XLM_BPE_CODES, XLM_BPE_VOCAB],
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)

  def __enter__(self):
    self.open()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def word2bpe(self, word):
    if self.proc is None:
      raise ConnectionError("open() not called")
    self.proc.stdin.write(('%s\n' % word).encode())
    self.proc.stdin.flush()
    result = self.proc.stdout.readline()
    return result.decode().strip()

  def close(self):
    if self.proc is None:
      return
    self.proc.stdin.close()
    self.proc.terminate()
    self.proc.wait(timeout=0.2)


