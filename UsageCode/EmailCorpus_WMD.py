###
"""
Written By: Samaksh (Avi) Goyal sagoyal@stanford.edu / mailsamakshgoyal@gmail.com

This file creates a word embedding model using all emails within the company, then uses Word Mover Distance between applicants
to find their "linguistic similarity score"

Run Command: python EmailCorpus_WMD.py

"""
###

import numpy as np
import multiprocessing
import pyreadr
import random
import spacy
import re

from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from itertools import compress
from time import time


def loadData():
    # OriginalRData
    result = pyreadr.read_r("jobvite_1_2_merged_anonymized.RData")  # output: odict_keys(['anon'])
    merged_anonymized = result["anon"]
    #print(merged_anonymized.shape)

    # Remove Duplicates, if someone was hired for one job and rejected for another keep only hired:
    merged_anonymized.sort_values(by=['Jobvite.ID', 'Hired'], ascending=[True, False], inplace=True)
    merged_anonymized.drop_duplicates(subset='Jobvite.ID', keep='first', inplace=True)
    #print(merged_anonymized.shape)

    return merged_anonymized


def textClean(Q1_Corpus, Q2_Corpus, Q3_Corpus):
    # Remove stopwords, numbers, punctuation etc.
    nlp = spacy.load('en', disable=['ner', 'parser'])

    # https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial

    def cleaning(doc):
        txt = [token.lemma_ for token in doc if not token.is_stop]
        if len(txt) > 2:
            return ' '.join(txt)

    brief_cleaning_Q1 = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in Q1_Corpus)
    brief_cleaning_Q2 = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in Q2_Corpus)
    brief_cleaning_Q3 = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in Q3_Corpus)

    t = time()

    txt_Q1 = [cleaning(doc) for doc in nlp.pipe(brief_cleaning_Q1, batch_size=5000, n_threads=-1)]
    txt_Q2 = [cleaning(doc) for doc in nlp.pipe(brief_cleaning_Q2, batch_size=5000, n_threads=-1)]
    txt_Q3 = [cleaning(doc) for doc in nlp.pipe(brief_cleaning_Q3, batch_size=5000, n_threads=-1)]

    print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

    return(txt_Q1, txt_Q2, txt_Q3)


def lineSplit(line):

    return line.split(" ")

def createEmbeddingSpace(filename):
    # you need to remake key common phrases...
    # "new york" should really be "new_york" as a collective since "new" and "york" have different meanings
    # if they are used together vs separately

    # https://stackoverflow.com/questions/35716121/how-to-extract-phrases-from-corpus-using-gensim

    #sentencesAll = []
    with open(filename, 'r') as f:

        sentencesAll = [line.split(" ") for line in f if line!= None]

    #takes about ~10 min
    random.shuffle(sentencesAll)

    phrases = Phrases(sentencesAll, min_count=1, threshold=2, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[sentencesAll]

    print(len(sentences)) #15,786,808
    print(sentences[0])

    # Building and Training the Model
    cores = multiprocessing.cpu_count()

    # I removed min_count... idk how to see which we not used
    w2v_model = Word2Vec(window=6,
                         size=100,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=cores - 1)

    t = time()

    w2v_model.build_vocab(sentences, progress_per=10000)

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2))) #6.71 mins

    t = time()

    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    print("Sentence[0]: in embedding Model {}".format(sentences[0]))
    print("Sentence[1]: in embedding Model {}".format(sentences[1]))
    print("Similarity is: {}".format(w2v_model.wv.wmdistance(sentences[0], sentences[1])))

    return w2v_model

def qAnalysis(Q_Corpus_Hired, txt_Q):

    sent = []
    keep_index = []
    sent_index = []

    for hired, keep, row in zip(Q_Corpus_Hired,range(len(Q_Corpus_Hired)), txt_Q):
        if row != None:
            sent.append(row)
            keep_index.append(keep)
            sent_index.append(hired)

    return(sent, keep_index, sent_index)


def updatedWMD(sentence_pair):

    score = 0

    for pair in sentence_pair:
        score += w2v_model.wv.wmdistance(pair[0],pair[1]) / len(pair[0].split())

    score = score/len(sentence_pair)

    return score

def findSimilarityScore(sent_Q, hired_Q):

    t = time()

    all_sentence_pairs = []
    for applicant in enumerate(sent_Q):
        sentence_pair = [(applicant[1], hired_applicant) for hired_applicant in hired_Q]
        all_sentence_pairs.append(sentence_pair)

    print('Time to find all sentence pairs: {} mins'.format(round((time() - t) / 60, 2)))

    print("...Done making all Sentence Pairs...")

    t = time()
    pool = multiprocessing.Pool(processes=10)
    all_similarity_score = pool.map(func=updatedWMD, iterable=all_sentence_pairs)
    print('Time to finish Pool: {} mins'.format(round((time() - t) / 60, 2)))

    pool.close()
    pool.join()

    return all_similarity_score

w2v_model = None

if __name__ == '__main__':
    merged_anonymized = loadData()
    print("...Data is Loaded...\n")

    #Extract only the answers for Q1, Q2, Q3
    Q1_Corpus = merged_anonymized[merged_anonymized['Q1']!=""]['Q1']
    Q2_Corpus = merged_anonymized[merged_anonymized['Q2']!=""]['Q2']
    Q3_Corpus = merged_anonymized[merged_anonymized['Q3']!=""]['Q3']


    (txt_Q1, txt_Q2, txt_Q3) = textClean(Q1_Corpus, Q2_Corpus, Q3_Corpus)
    print("...Data is Cleaned...\n")

    filename = r"allSentencescleaned.txt"
    w2v_model = createEmbeddingSpace(filename)
    print("...Embedding Space is made...\n")

    ####################################################################################################
    #Q1 Analysis
    Q1_Corpus_Hired = merged_anonymized[merged_anonymized['Q1'] != ""]['Hired'] == 1
    (sent_Q1,keep_index_Q1, sent_Q1_index) = qAnalysis(Q1_Corpus_Hired, txt_Q1)
    np.savetxt('np_q1_keep_Q1_index_Email.txt', np.array(keep_index_Q1))
    hired_Q1_index = list(compress(range(len(sent_Q1_index)), sent_Q1_index))
    hired_Q1 = list(compress(sent_Q1, sent_Q1_index))

    print("...Q1 Analysis ready to start...\n")
    q1_similarity_score = findSimilarityScore(sent_Q1, hired_Q1)
    print("...Q1 Similarity Score was found...\n")
    np_q1_similarity_score = np.array(q1_similarity_score)
    np.savetxt('np_q1_similarity_score_Email_wmd.txt', np_q1_similarity_score)

    ####################################################################################################
    #Q2 Analysis
    Q2_Corpus_Hired = merged_anonymized[merged_anonymized['Q2'] != ""]['Hired'] == 1
    (sent_Q2,keep_index_Q2, sent_Q2_index) = qAnalysis(Q2_Corpus_Hired, txt_Q2)
    np.savetxt('np_q2_keep_Q2_index_Email.txt', np.array(keep_index_Q2))
    hired_Q2_index = list(compress(range(len(sent_Q2_index)), sent_Q2_index))
    hired_Q2 = list(compress(sent_Q2, sent_Q2_index))

    print("...Q2 Analysis ready to start...\n")
    q2_similarity_score = findSimilarityScore(sent_Q2, hired_Q2)
    print("...Q2 Similarity Score was found...\n")
    np_q2_similarity_score = np.array(q2_similarity_score)
    np.savetxt('np_q2_similarity_score_Email_wmd.txt', np_q2_similarity_score)

    ####################################################################################################
    #Q3 Analysis
    Q3_Corpus_Hired = merged_anonymized[merged_anonymized['Q3'] != ""]['Hired'] == 1
    (sent_Q3,keep_index_Q3, sent_Q3_index) = qAnalysis(Q3_Corpus_Hired, txt_Q3)
    np.savetxt('np_q3_keep_Q3_index_Email.txt', np.array(keep_index_Q3))
    hired_Q3_index = list(compress(range(len(sent_Q3_index)), sent_Q3_index))
    hired_Q3 = list(compress(sent_Q3, sent_Q3_index))

    print("...Q3 Analysis ready to start...\n")
    q3_similarity_score = findSimilarityScore(sent_Q3, hired_Q3)
    print("...Q3 Similarity Score was found...\n")
    np_q3_similarity_score = np.array(q3_similarity_score)
    np.savetxt('np_q3_similarity_score_Email_wmd.txt', np_q3_similarity_score)
