import os
import pickle
import time
import unicodedata
import numpy as np
from src import config as c

from collections import Counter
from itertools import chain, compress
from functools import partial
from multiprocessing.pool import Pool
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException

from src.experiment.evaluate import evaluate_map
from src.experiment.index import create_index
from src.model.text2vec import clean, tokenize, lookup
from src.util.timer import Timer as PPrintTimer


_timer = PPrintTimer()
def run_unigram_lm(query_lang: str, doc_lang: str, experiment_data: tuple, processes=40, timer=None):
    """
    Builds a unigram language model (dirichlet smoothing) and
    :param query_lang: query language, e.g. EN
    :param doc_lang: document language, e.g. DE
    :param experiment_data: doc-ids, documents, query-ids, queries, relevance assessments
    :param processes: number of processes (multiprocessing)
    :param timer:
    :return:
    """
    if not timer:
        timer = PPrintTimer()
    else:
        timer = _timer

    _, qlang_long = query_lang
    _, dlang_long = doc_lang
    doc_ids, documents, query_ids, queries, relass = experiment_data

    with Pool(processes=processes) as pool:
        print("Start preprocessing data %s" % timer.pprint_lap())
        clean_to_lower = partial(clean, to_lower=True)

        tokenize_query_language = partial(tokenize, language=qlang_long, exclude_digits=True)
        queries = pool.map(clean_to_lower, queries)
        queries = pool.map(tokenize_query_language, queries)
        print("Queries preprocessed %s" % timer.pprint_lap())

        documents = pool.map(clean_to_lower, documents)
        tokenize_doc_language = partial(tokenize, language=dlang_long, exclude_digits=True)
        documents = pool.map(tokenize_doc_language, documents)
        print("Documents preprocessed %s" % (timer.pprint_lap()))

        # word frequency distribution per document
        doc_freq_distributions = pool.map(_count_words, documents)

    # build document language models
    doc_prob_distributions = []
    for elem in doc_freq_distributions:
        doc_len = sum(elem.values())
        doc_prob_distributions.append({k: v/doc_len for k,v in elem.items()})
    print("Document conditional counts collected %s" % timer.pprint_lap())

    # build collection language model
    collection_freq_distribution = Counter()
    for document in doc_freq_distributions:
        collection_freq_distribution.update(document)  # { token: frequency }
    collection_freq_distribution = dict(collection_freq_distribution)
    colletion_len = sum(collection_freq_distribution.values())
    collection_prob_distribution = {term: freq/colletion_len for term, freq in collection_freq_distribution.items()}
    print("Marginal counts collected %s" % timer.pprint_lap())

    # Interpolation/Smoothing weights
    doclen_list = [sum(doc.values()) for doc in doc_freq_distributions]
    doclen_set = set(doclen_list)
    doclen2lambda = {doclen: doclen / (doclen + 1000) for doclen in doclen_set}
    lambdas = np.array([doclen2lambda[sum(doc_freq.values())] for doc_freq in doc_freq_distributions])
    lambdas = np.expand_dims(lambdas, 1)

    np.random.seed(10)
    results = []
    print("start evaluation %s" % timer.pprint_lap())
    print_every_x = 50 if len(queries) > 60 else 10
    queries_without_relevant_docs = []
    for i, query in enumerate(queries, 1):
        query_id = query_ids[i - 1]

        # filter out terms that occur nowhere in the collection
        query = [term for term in query if term in collection_freq_distribution]
        any_lexical_overlap_with_collection = len(query) > 0

        if query_id in relass and any_lexical_overlap_with_collection:
            ranking = rank_unigram_lm(collection_LM=collection_prob_distribution,
                                      document_LMs=doc_prob_distributions,
                                      lambdas=lambdas,
                                      query=query)
            results.append(ranking)
        else:
            random_ranking = np.random.permutation(len(documents))
            results.append(random_ranking)  # query without relevant documents is not fired
            queries_without_relevant_docs.append(query_id)
        if i % print_every_x == 0:
            print("%s  queries processed (%s)" % (i, timer.pprint_lap()))

    query2ranking = {}
    for i, query_id in enumerate(query_ids):
        query2ranking[query_id] = [doc_ids[_id] for _id in results[i]]

    mean_average_precision, pvalue = evaluate_map(query2ranking=query2ranking, relass=relass)
    print("unigram lm done %s" % timer.pprint_lap())
    return mean_average_precision


def rank_unigram_lm(collection_LM, document_LMs, lambdas, query):
    random_ranking = np.random.permutation(len(document_LMs))
    # query should contain duplicate words if applicable
    doc_log_probabilities = score_unigram_lm(collection_LM, lambdas, document_LMs, query)
    doc_log_probabilities *= -1  # sort descending
    # lexsort: same as np.argsort(doc_log_probabilities) but shuffle docs with equal scores
    ranking = np.lexsort((random_ranking, doc_log_probabilities))
    return ranking


def score_unigram_lm(collection_LM, lambdas, document_LMs, query):
    collection_term_probs = np.array([collection_LM.get(term, 0) for term in query])
    doc_term_probs = np.array([[doc_prob.get(term, 0) for term in query] for doc_prob in document_LMs])
    doc_log_probabilities = np.log(lambdas * doc_term_probs + (1 - lambdas) * collection_term_probs).sum(axis=1)
    return doc_log_probabilities


def run_googletranslate_translation(query_lang, doc_lang, experiment_data):
    # unpacking values
    qlang_short, qlang_long = query_lang
    dlang_short, dlang_long = doc_lang
    doc_ids, documents, query_ids, queries, relass = experiment_data

    def translate(text, src, tgt, sleeptime):
        driver = webdriver.PhantomJS(c.PATH_PHANTOM_JS)
        gquery = "https://translate.google.com/#%s/%s/%s" % (src, tgt, text)
        driver.get(gquery)
        # res = driver.find_element_by_xpath("//span[@class='tlid-translation translation']").text
        res = driver.find_element_by_xpath("//div[@class='zkZ4Kc dHeVVb']").get_attribute("data-text")
        time.sleep(sleeptime)
        return res

    file = c.GTRANSLATE_CACHE + "%s_translated_to_%s.pickle" % (qlang_short, dlang_short)
    if os.path.exists(file):
        with open(file, "rb") as f:
            translated_queries = pickle.load(f)
    else:
        translated_queries = []
        for query in queries:
            translated_query = None
            sleep = np.random.randint(5, 20)  # seconds
            while translated_query is None:
                try:
                    translated_query = translate(query, qlang_short, dlang_short, sleep)
                except NoSuchElementException:
                    sleep += sleep
            translated_queries.append(translated_query)
            if len(translated_queries) % 5 == 0:
                print("%s queries translated (from %s to %s, in %s)" % (str(len(translated_queries)), qlang_short,
                                                                        dlang_short, _timer.pprint_lap()))
        os.makedirs(c.GTRANSLATE_CACHE, exist_ok=True)
        with open(file, "wb") as f:
            pickle.dump(translated_queries, f)

    print("Queries translated, now running Unigram LM %s" % _timer.pprint_lap())
    new_experiment_data = doc_ids, documents, query_ids, translated_queries, relass
    return run_unigram_lm(query_lang=query_lang, doc_lang=doc_lang, experiment_data=new_experiment_data)


def run_wordbyword_translation(query_lang, doc_lang, experiment_data, initialized_embeddings, processes=40, emb_size=300):
    # unpacking values
    qlang_short, qlang_long = query_lang
    dlang_short, dlang_long = doc_lang
    doc_ids, documents, query_ids, queries, relass = experiment_data
    embeddings = initialized_embeddings

    with Pool(processes=processes) as pool:
        queries = list(pool.map(clean, queries))
        tokenize_language = partial(tokenize, language=qlang_long)
        # queries_tokenized = list(map(tokenize_language, queries))
        queries_tokenized = list()
        for tokenized_query in pool.imap(tokenize_language, queries):
            queries_tokenized.append(tokenized_query)
    unique_query_terms = list(set(chain(*queries_tokenized)))

    doc_language_vocabulary = [k for k, v in embeddings.lang_vocabularies[dlang_long].items()]
    doc_language_embeddings = embeddings.lang_embeddings[dlang_long]

    print("Vocabulary extraced %s" % _timer.pprint_lap())
    index, quantizer = create_index(doc_language_embeddings)
    search_vecs = []
    zero_vec = np.zeros(emb_size, dtype=np.float32)
    keep_words_as_is = set()
    for unique_query_term in unique_query_terms:
        lookedup_word, vec = lookup(unique_query_term, qlang_long, embedding_lookup=embeddings)
        if vec is not None:
            search_vecs.append(vec)
        else:
            keep_words_as_is.add(lookedup_word)
            search_vecs.append(zero_vec)

    search_vecs = np.array(search_vecs, dtype=np.float32)
    non_zeros = np.all(search_vecs != 0, axis=1)
    # Word embeddings of english (query language) words
    search_vecs = search_vecs[non_zeros]
    unique_query_terms = list(compress(unique_query_terms, non_zeros))

    nearest_neighbors = 1
    # search in index of (document language) embeddings/words
    _, I = index.search(search_vecs, nearest_neighbors)
    print("Nearest neighbors / translation mapping computed %s" % _timer.pprint_lap())

    nearest_neighbors_of_unique_query_terms = [doc_language_vocabulary[nearest_neighbor.tolist()[0]] for
                                               nearest_neighbor in I]
    nearest_neighbor_mapping = dict(zip(unique_query_terms, nearest_neighbors_of_unique_query_terms))

    def translate(query):
        translation = []
        for query_term in query:
            if query_term in keep_words_as_is:
                translation.append(query_term)
                continue

            translated_query_term = None
            if query_term in nearest_neighbor_mapping:
                translated_query_term = nearest_neighbor_mapping[query_term]
            elif query_term.lower() in nearest_neighbor_mapping:
                translated_query_term = nearest_neighbor_mapping[query_term.lower()]

            if translated_query_term is None:
                no_special_chars = ''.join(
                    (char for char in unicodedata.normalize('NFD', query_term) if unicodedata.category(char) != 'Mn'))
                if no_special_chars in nearest_neighbor_mapping:
                    translated_query_term = nearest_neighbor_mapping[no_special_chars]
                elif no_special_chars.lower() in nearest_neighbor_mapping:
                    translated_query_term = nearest_neighbor_mapping[no_special_chars.lower]
            translation.append(translated_query_term)
        return ' '.join([word for word in translation if word is not None])

    translated_queries = list(map(translate, queries_tokenized))
    print("Queries translated, now running Unigram LM %s" % _timer.pprint_lap())
    new_experiment_data = doc_ids, documents, query_ids, translated_queries, relass
    return run_unigram_lm(query_lang=query_lang, doc_lang=doc_lang, experiment_data=new_experiment_data,
                          processes=processes)


def _count_words(document):
    return dict(Counter(document))
