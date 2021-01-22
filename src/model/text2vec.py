import copy
import re
import string
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from multiprocessing import Pool
from functools import partial
from itertools import chain
from collections import Counter
import tqdm
import unicodedata

regex = re.compile('[%s]' % re.escape(string.punctuation))


def clean(_str, to_lower=True, strip_accents=False, strip_punctuation=True):
    """
    Cleans string from newlines and punctuation characters
    :param _str:
    :param strip_accents:
    :param to_lower:
    :param strip_punctuation:
    :return:
    """
    if strip_accents:
        _str = run_strip_accents(_str)

    if to_lower:
        _str = _str.lower()
        # _str = _str.replace("find reports on"," ")
        # _str = _str.replace("find documents", " ")

    if _str is not None and strip_punctuation:
        _str = _str.replace("\n", " ").replace("\r", " ")
        return regex.sub(' ', _str)
    else:
        return _str


def run_strip_accents(text):
  """
  Strips accents from a piece of text.
  """
  text = unicodedata.normalize("NFD", text)
  output = []
  for char in text:
    cat = unicodedata.category(char)
    if cat == "Mn":
      continue
    output.append(char)
  return "".join(output)



def sent_split(text, language):
    return sent_tokenize(text, language=language)


def tokenize(text, language, exclude_digits=False):
    """
    Call first clean then this function.
    :param exclude_digits: whether include or exclude digits
    :param text: string to be tokenized
    :param language: language flag for retrieving stop words
    :return:
    """
    from nltk.corpus import stopwords
    stopwords = set(stopwords.words(language))
    punctuation = set(string.punctuation)
    tokens = []
    for token in word_tokenize(text, language=language):
        if token not in stopwords and token.lower() not in stopwords and token not in punctuation and len(token) > 1:
            if exclude_digits:
                if not any(t.isdigit() for t in token):
                    tokens.append(token)
            else:
                tokens.append(token)
    return tokens



def lookup(token, language, embedding_lookup=None):
    """
    First try to lookup embedding from word as it appears in text, then its lowercase
    version, then its version with accents stripped, then its its version with accents
    stripped and lowercased letters.

    :param token: term to be looked up
    :param language: language in which term occurs
    :param embedding_lookup: embedding helper used for lookup
    :return:
    """
    global embeddings
    tmp = embedding_lookup if embedding_lookup is not None else embeddings

    if type(tmp) == dict:
        local_lookup = lambda lang, tok: tmp[lang].get(tok, None)
    else:
        local_lookup = lambda lang, tok: tmp.get_vector(lang, tok)

    word_vector = local_lookup(language, token)
    word = token
    if word_vector is None:
        word_vector = local_lookup(language, token.lower())
        word = token.lower()
    if word_vector is None:
        no_special_chars = ''.join((c for c in unicodedata.normalize('NFD', token) if unicodedata.category(c) != 'Mn'))
        word_vector = local_lookup(language, no_special_chars)
        word = no_special_chars
    if word_vector is None:
        word_vector = local_lookup(language, no_special_chars.lower())
        word = no_special_chars.lower()

    return word, word_vector


def text2vec_bigram(text, language='', dim=300):
    """
    Document aggregation method presented in http://www.aclweb.org/anthology/P14-1006

    :param text:
    :param language:
    :param dim:
    :return:
    """
    _id, txt = text
    text = clean(txt)
    zero_vec = np.zeros(dim)
    tokens = tokenize(text, language=language)

    bigram_vectors = []
    for word_1, word_2 in nltk.bigrams(tokens):
        _, emb_1 = lookup(word_1, language)
        if emb_1 is None:
            emb_1 = zero_vec
        _, emb_2 = lookup(word_2, language)
        if emb_2 is None:
            emb_2 = zero_vec
        bigram_vectors.append(np.tanh(emb_1 + emb_2))

    document_vector = np.sum(bigram_vectors, 0) if len(bigram_vectors) > 0 else zero_vec
    if not np.array_equal(document_vector, zero_vec):
        try:
                document_vector /= np.linalg.norm(document_vector, 2)
        except Exception as e:  # TODO: fix broad exception
            document_vector = zero_vec
    return document_vector, [], [], 0


def text2vec_idf_sum(text, language='', dim=300, to_lower=True):
    """
    This function is used used only as a symbol (for a unified interface). The idf-scaling
    occurs in the caller: create_text_representations.

    :param text: query / document
    :param language: language of text
    :param dim: dimensionality of word embeddings
    :param to_lower: apply lowercasing
    :return:
    """
    return text2vec_sum(text, language=language, dim=dim, to_lower=to_lower)


def text2vec_sum(text, language='', dim=300, to_lower=True):
    """
    Transforms text into vector representation by embedding lookup on shared/bilingual embedding
    space. The text is represented as a sum of the word embeddings.

    :param language: language of text
    :param dim: dimensionality of word embeddings
    :param text: query / document
    :param to_lower: apply lowercasing
    :return: unit vector of the sum
    """
    _id, txt = text
    text = clean(txt, to_lower=to_lower)

    unknown_words = []
    word_vectors = []
    zero_vec = np.zeros(dim)

    _all = 0
    _unique = set()
    for token in tokenize(text, language=language):
        _all += 1
        _unique.add(token)

        _, word_vector = lookup(token, language)
        if word_vector is None:  # tv-wereld, oost-duitsland (split and lookup each token)
            word_vector = zero_vec
            if not any(t.isdigit() for t in token):
                unknown_words.append(token)
        elif not word_vector[0] > 500:
            pass
        word_vectors.append(word_vector)

    document_vector = np.sum(word_vectors, 0)
    if not np.array_equal(document_vector, zero_vec):
        try:
            document_vector /= np.linalg.norm(document_vector, 2)
        except:
            document_vector = zero_vec
    return document_vector, unknown_words, _all, len(_unique)


def _map_func(id_text, language, normalize=True, embedding_lookup=None, to_lower=True):
    """
    Helper function for computing IDF weights, text -> list of words

    :param id_text: (document id, text) tuple
    :param language: text language
    :return:
    """
    _id, doc = id_text
    doc = tokenize(clean(doc, to_lower=to_lower), language=language)
    # result = list(set(doc))

    result = []
    for unnormalized_word in list(set(doc)):
        if normalize:
            normalized_word, _  = lookup(unnormalized_word, language, embedding_lookup)
            result.append(normalized_word)
        else:
            result.append(unnormalized_word.strip())

    return result


def compute_idf_weights(text, language, processes, normalize=True, embedding_lookup=None, return_doc_freqs=False,
                        to_lower=True):
    """
    Returns a mapping { term: IDF_term }

    :param text: list of documents in corpus
    :param language: corpus language
    :param processes: paralellization parameter
    :param normalize: whether to take word as is or normalize word-form
    :param to_lower: apply lowercasing
    :param embedding_lookup: lookup table
    :param return_doc_freqs: whether to return idf-mapping only or tuple (term-to-idf mapping, term-to-docfreq mapping)
    :return:
    """
    if processes > 1:
        global embeddings
        embeddings = embedding_lookup
        with Pool(processes=processes) as pool:
            _map_func_language = partial(_map_func, language=language, normalize=normalize, to_lower=to_lower)
            # each occurrence of a word results from one document
            words = []
            for tmp in tqdm.tqdm(pool.imap(_map_func_language, text), total=len(text)):
                words.append(tmp)
            # words = pool.map(_map_func_language, text)
    else:
        words = [_map_func(line, language=language, normalize=normalize, to_lower=to_lower) for line in text]

    collection_size = len(text)
    flat_words = list(chain(*words))
    doc_frequencies = dict(Counter(flat_words))
    idf_mapping = {term: np.log(collection_size / doc_frequency) for term, doc_frequency in doc_frequencies.items()}
    idf_mapping['AVG_IDF'] = sum(list(idf_mapping.values())) / len(idf_mapping)
    result = idf_mapping if not return_doc_freqs else (idf_mapping, doc_frequencies)
    return result


def create_text_representations(language, id_text, emb, aggregation=text2vec_sum, processes=40,
                                idf_weighing=False, emb_dim=300, to_lower=True):
    """
    Runs a text2vec method in parallel to transform documents to document vectors. It
    requires a global embedding variable to be created first.

    :param language: used for the look-up table "embeddings"
    :param id_text: (doc_id, document_tokens)
    :param aggregation: textvec variant
    :param emb: embedding
    :param processes: number processes to run in parallel
    :param idf_weighing: whether to rescale word embeddings with the words idf
    :param emb_dim: dimensionality of embeddings
    :param to_lower: apply lowercasing
    :return:
    """
    global embeddings
    embeddings = emb

    id_text = list(id_text)
    if idf_weighing:
        # We modify the embedding object and want to reuse the original one for subsequent experiments
        embeddings = copy.deepcopy(emb)
        # compute IDF weights
        idf_weights = compute_idf_weights(id_text, language, processes, embedding_lookup=embeddings, to_lower=to_lower)
        print("idf weights computed")

        for key in embeddings.lang_vocabularies[language].keys():
            try:
                idf = idf_weights[key]
                old_embedding = embeddings.get_vector(lang=language, word=key)
                rescaled_embedding = old_embedding * idf
                embeddings.set_vector(lang=language, word=key, vector=rescaled_embedding)
            except:
                pass
        print("...old weights idf-rescaled")

    pool = Pool(processes=processes)
    partial_method = partial(aggregation, language=language, dim=emb_dim, to_lower=to_lower)
    # results = pool.starmap(method, id_text)
    results = []
    for result in tqdm.tqdm(pool.imap(partial_method, id_text), total=len(id_text)):
        results.append(result)
    pool.close()
    pool.join()
    # unknown_words = list(chain(*[tmp[1] for tmp in results]))
    # filtered_unknown_words = [token for token in unknown_words if lookup(token, "procb_de")[1] is not None]
    return np.array([result[0] for result in results], dtype=np.float32)
