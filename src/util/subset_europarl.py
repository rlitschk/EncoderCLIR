import random
from src.dataloaders.extractors import load_txt_data

path = "/path/to/europarl/"


def sample_europarl(src_path, tgt_path):
    query_lines = load_txt_data(src_path)
    document_lines = load_txt_data(tgt_path)

    assert len(query_lines) == len(document_lines)

    random.seed(432)
    random_permutation = list(range(len(query_lines)))
    random.shuffle(random_permutation)
    query_indices = random_permutation[:5000]
    document_indices = random_permutation[:200000]

    shuffled_queries = [query_lines[idx] for idx in query_indices]
    shuffled_docs = [document_lines[idx] for idx in document_indices]

    with open(src_path + ".queries", "w") as f:
        f.writelines(shuffled_queries)

    with open(tgt_path + ".documents", "w") as f:
        f.writelines(shuffled_docs)


sample_europarl(path + "de-fi.txt/Europarl.de-fi.de", path + "de-fi.txt/Europarl.de-fi.fi")  # de - fi
print("Done de-fi")
sample_europarl(path + "de-it.txt/Europarl.de-it.de", path + "de-it.txt/Europarl.de-it.it")  # de - it
print("Done de-it")

sample_europarl(path + "de-en.txt/Europarl.de-en.en", path + "de-en.txt/Europarl.de-en.de")  # en - de
print("Done de-en")
sample_europarl(path + "en-fi.txt/Europarl.en-fi.en", path + "en-fi.txt/Europarl.en-fi.fi")  # en - fi
print("Done en-fi")
sample_europarl(path + "en-it.txt/Europarl.en-it.en", path + "en-it.txt/Europarl.en-it.it")  # en - it
print("Done en-it")

sample_europarl(path + "fi-it.txt/Europarl.fi-it.fi", path + "fi-it.txt/Europarl.fi-it.it")  # fi - it
print("Done fi-it")
