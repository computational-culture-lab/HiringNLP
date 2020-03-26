# HiringNLP

Greatings Reader! Below is an extension of Sarah's work by adding new NLP models in an attempt to find new similarity metrics.

## Linguistic Similarity

To measure Linguistic (Semantic) Similarity between applicants, we tried
several methods of modeling response similarities \[1\]. Below is the
general procedure:

<span>**1.**</span> Construct pairwise similarities between all
applicant responses (based on some similarity function \(sim\)).
Resulting in a matrix as such:

```math
\[S = \begin{bmatrix} 
0 & sim(\theta)_{12} & \dots & sim(\theta)_{1n} \\
sim(\theta)_{21} & 0 & \dots & \vdots \\
\vdots & \vdots & \ddots & \vdots \\
sim(\theta)_{n1} & \dots & \dots & 0
\end{bmatrix}\]
```

<span>**2.**</span> Construct an independent pre-hire fit measure,
\(S_H\), by multiplying the similarity matrix, \(S\), by a vector of
binary hiring outcomes, \(H= [0,1]\). This resulted in a variable
measuring the degree to which each individual’s language was similar to
the language used by the group of hired individuals (excluding himself
or herself when appropriate).

\[H = \begin{bmatrix}
h_1 \\ h_2 \\ \vdots \\ h_n
\end{bmatrix},
\tab
%
S_H = S*h = \begin{bmatrix} 
0 & sim(\theta)_{12} & \dots & sim(\theta)_{1n} \\
sim(\theta)_{21} & 0 & \dots & \vdots \\
\vdots & \vdots & \ddots & \vdots \\
sim(\theta)_{n1} & \dots & \dots & 0
\end{bmatrix}
%
\begin{bmatrix}
  h_1 \\ h_2 \\ \vdots \\ h_n
\end{bmatrix}\]

For each method of modeling response similarity we have different
procedures:

### TF-IDF

For each applicant response a Term Frequency vector was made and then
cosine similarities (\(sim = cos\)) were computed between all applicant
responses. This will be our baseline against which to compare all latter
embedding models.

### Word2Vec Average Word

Either all essay responses or firm emails were used as the training
corpus to create CBOW Word2Vec embeddings. Then for each response, we
found the average word embedding by iterating over all words in the
response to find centroid. Then cosine similarities (\(sim = cos\)) were
computed between all applicant average responses.

### Word2Vec Word Mover Distance

Either all essay responses or firm emails were used as the training
corpus to create CBOW Word2Vec embeddings. Then for each response we
used Word Mover distances between applicant responses to model sentence
similarity (\(sim = WMD\)).

### Universal Sentence Encoder

Pre-trained fixed length sentence embedding from Google were downloaded
and retrieved for each applicant response. Then cosine similarities
(\(sim = cos\)) were computed between all USE sentence encoding of
applicant responses.

### Sentence BERT

Pre-trained fixed length sentence embedding from Siamese BERT network
were downloaded and retrieved for each applicant response. Then cosine
similarities (\(sim = cos\)) were computed between all SBERT sentence
encoding of appl

1.  I would like to mention that this part of the paper is my original
    contribution to the project. I researched, found and developed ways
    of using existing NLP work based on semantic textual similarity to
    apply to this project.
