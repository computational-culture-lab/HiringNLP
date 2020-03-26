###
"""
Written By: Samaksh (Avi) Goyal sagoyal@stanford.edu / mailsamakshgoyal@gmail.com

This file uses pre-trained Siamese Bert Network embeddings on applicant sentences, and uses paired cosine similairty to
to find their "linguistic similarity score"

Run Command: python Sent_Bert.py

"""
###

import numpy as np
from sentence_transformers import SentenceTransformer
from itertools import compress
from time import time
from scipy import spatial
import multiprocessing
import pyreadr

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

def qAnalysis(Q_Corpus_Hired, txt_Q):

    sent = []
    keep_index = [] #index of which responses are !=None in Q1
    sent_index = [] #index of which responses were from hired applicants in Q1

    model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
    sent_embeddings = model.encode(txt_Q)
    # print(txt_Q[0:5])
    # print(sent_embeddings[0:5])

    for hired, keep, row in zip(Q_Corpus_Hired,range(len(Q_Corpus_Hired)), sent_embeddings):
        #if row != None:
        sent.append(row)
        keep_index.append(keep)
        sent_index.append(hired)

    return(sent, keep_index, sent_index)


def SentBertscore(sentence_pair):
    # Distance = 1 - Similarity
    # distance: 10 very distant
    # similarity: 10 very similar

    similarity = 0

    for pair in sentence_pair:
        similarity += (1 - spatial.distance.cosine(pair[0], pair[1]))

    similarity = similarity/len(sentence_pair)

    return similarity

def findSentBertSimilarityScore(sent_Q, hired_Q):

    t = time()

    all_sentence_pairs = []
    for applicant in enumerate(sent_Q):
        sentence_pair = [(applicant[1], hired_applicant) for hired_applicant in hired_Q]
        all_sentence_pairs.append(sentence_pair)

    print('Time to find all sentence pairs: {} mins'.format(round((time() - t) / 60, 2)))

    print("...Done making all Sentence Pairs...")

    t = time()
    pool = multiprocessing.Pool(processes=30)
    all_similarity_score = pool.map(func=SentBertscore, iterable=all_sentence_pairs)
    print('Time to finish Pool: {} mins'.format(round((time() - t) / 60, 2)))

    pool.close()
    pool.join()

    return all_similarity_score

if __name__ == '__main__':
    merged_anonymized = loadData()
    print("...Data is Loaded...\n")

    #Extract only the answers for Q1, Q2, Q3
    Q1_Corpus = merged_anonymized[merged_anonymized['Q1']!=""]['Q1']
    Q2_Corpus = merged_anonymized[merged_anonymized['Q2']!=""]['Q2']
    Q3_Corpus = merged_anonymized[merged_anonymized['Q3']!=""]['Q3']

    ####################################################################################################
    # Q1 Analysis
    Q1_Corpus_Hired = merged_anonymized[merged_anonymized['Q1'] != ""]['Hired']==1 # index ___ True/False
    (sent_Q1,keep_index_Q1, sent_Q1_index) = qAnalysis(Q1_Corpus_Hired, Q1_Corpus.tolist())
    hired_Q1_index = list(compress(range(len(sent_Q1_index)), sent_Q1_index)) #index of hired application from Q1
    hired_Q1 = list(compress(sent_Q1, sent_Q1_index)) #sentence only from hired applicants

    print("...Q1 Analysis ready to start...\n")
    q1_Sent_BERT_similarity_score = findSentBertSimilarityScore(sent_Q1, hired_Q1)
    print("...Q1 USE Similarity Score was found...\n")
    q1_Sent_BERT_similarity_score = np.array(q1_Sent_BERT_similarity_score)
    np.savetxt('np_q1_SENT_BERT_similarity_score.txt', q1_Sent_BERT_similarity_score)

    ####################################################################################################
    # Q2 Analysis
    Q2_Corpus_Hired = merged_anonymized[merged_anonymized['Q2'] != ""]['Hired']==1 # index ___ True/False
    (sent_Q2,keep_index_Q2, sent_Q2_index) = qAnalysis(Q2_Corpus_Hired, Q2_Corpus.tolist())
    hired_Q2_index = list(compress(range(len(sent_Q2_index)), sent_Q2_index)) #index of hired application from Q1
    hired_Q2 = list(compress(sent_Q2, sent_Q2_index)) #sentence only from hired applicants

    print("...Q2 Analysis ready to start...\n")
    q2_Sent_BERT_similarity_score = findSentBertSimilarityScore(sent_Q2, hired_Q2)
    print("...Q2 USE Similarity Score was found...\n")
    q2_Sent_BERT_similarity_score = np.array(q2_Sent_BERT_similarity_score)
    np.savetxt('np_q2_SENT_BERT_similarity_score.txt', q2_Sent_BERT_similarity_score)

    ####################################################################################################
    # Q3 Analysis
    Q3_Corpus_Hired = merged_anonymized[merged_anonymized['Q3'] != ""]['Hired']==1 # index ___ True/False
    (sent_Q3,keep_index_Q3, sent_Q3_index) = qAnalysis(Q3_Corpus_Hired, Q3_Corpus.tolist())
    hired_Q3_index = list(compress(range(len(sent_Q3_index)), sent_Q3_index)) #index of hired application from Q1
    hired_Q3 = list(compress(sent_Q3, sent_Q3_index)) #sentence only from hired applicants

    print("...Q3 Analysis ready to start...\n")
    q3_Sent_BERT_similarity_score = findSentBertSimilarityScore(sent_Q3, hired_Q3)
    print("...Q3 USE Similarity Score was found...\n")
    q3_Sent_BERT_similarity_score = np.array(q3_Sent_BERT_similarity_score)
    np.savetxt('np_q3_SENT_BERT_similarity_score.txt', q3_Sent_BERT_similarity_score)

