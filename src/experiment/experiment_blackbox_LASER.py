import os

import numpy as np
import subprocess
import tempfile

from src.config import LASER_EMB
from src.experiment.evaluate import layer_wise_evaluation


def laser_blackbox_experiment(query_lang, doc_lang, experiment_data, encode_fn, directory="", batch_size=4, **kwargs):
    print("running LASER blackbox experiment")
    doc_ids, raw_documents, query_ids, raw_queries, relass = experiment_data
    tmp_ids = []
    tmp_docs = []
    for _id, doc in zip(doc_ids, raw_documents):
        if doc.strip() != "":
            tmp_ids.append(_id)
            tmp_docs.append(" ".join(doc.split()))#[:kwargs['maxlen']]))
    raw_documents = tmp_docs
    doc_ids = tmp_ids

    raw_documents = [" ".join(line.lower().split()) for line in raw_documents]
    raw_queries = [" ".join(line.lower().split()) for line in raw_queries]

    dim = 1024

    # tokenize
    tmp_query_embeddings_path = directory + "tmp_%s_%s_%s_queries.npy" % (query_lang, str(len(raw_queries)), kwargs["maxlen"])
    query_representations = _generate_laser_embeddings(query_lang, raw_queries, tmp_query_embeddings_path, dim, directory)
    print("query emb file: %s" % tmp_query_embeddings_path)


    tmp_doc_embeddings_path = directory + "tmp_%s_%s_%s_docs.npy" % (doc_lang, str(len(raw_documents)), kwargs["maxlen"])
    document_representations = _generate_laser_embeddings(doc_lang, raw_documents, tmp_doc_embeddings_path, dim, directory)
    print("doc emb file: %s" % tmp_doc_embeddings_path)

    lang_pair = query_lang + "-" + doc_lang
    result = layer_wise_evaluation(doc_ids, document_representations, query_ids, query_representations, relass,
                                   lang_pair=lang_pair, model_dir=directory)
    print("max-len: %s\tmap:%s" % (str(kwargs["maxlen"]), str(result[0])))
    return result


def _generate_laser_embeddings(language, raw_textlines, embeddings_path, dim, tmpfiledir):
    if not os.path.exists(embeddings_path):
        os.makedirs(tmpfiledir, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=tmpfiledir) as tmpfile:
            tmpfile.writelines([line + "\n" for line in raw_textlines])
            tmpfile.flush()
            tmpfilename = tmpfile.name
            subprocess.run(" ".join([LASER_EMB, tmpfilename, language,
                                     embeddings_path]),
                           env=dict(os.environ),
                           shell=True)
    embs = np.fromfile(embeddings_path, dtype=np.float32, count=-1)
    embs.resize(embs.shape[0] // dim, dim)
    return embs
