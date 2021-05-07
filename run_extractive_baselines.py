import argparse
import random
import json
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', dest='model_path', type=str, default=None,
                        help='Directory for bert model')
    parser.add_argument('--data_path', dest='data_path', type=str, default=None,
                        help='Directory for testing data')
    parser.add_argument('--output_path', dest='output_path', type=str, default=None,
                        help='Directory for saving output json file')
    parser.add_argument('--heuristic_type', dest='heuristic_type', type=str, default=None,
                        help='Heuristic baseline to use')
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=42,
                        help='Random Seed')
    parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=1,
                        help='Testing batch size')
    args = parser.parse_args()
    return args

def _set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = get_params()
    _set_random_seeds(args.random_seed)
    df = pd.read_json(args.data_path, orient='split')
    #df = df.head(100)

    questions = [q if q.endswith("?") else q+"?" for q in df.question]
    reviews = [r for r in df.passages]
    multiple_answers = [a for a in df.multiple_answers]

    questions = [q.lower() for q in questions]
    reviews = [[p.lower() for p in r] for r in reviews]
    multiple_answers = [[a.lower() for a in ans] for ans in multiple_answers]

    pred_answers = []
    if args.heuristic_type == 'tfidf':
        for i in range(len(questions)):
            review_concat = ''
            for j in reviews[i]:
                review_concat += j + '' 
            sent_list = sent_tokenize(review_concat)
            sent_list.insert(0, questions[i])

            tfidfvectoriser=TfidfVectorizer(max_features=100)
            tfidfvectoriser.fit(sent_list)
            tfidf_vectors=tfidfvectoriser.transform(sent_list)

            tfidf_vectors=tfidf_vectors.toarray()
            pairwise_similarities=euclidean_distances(tfidf_vectors)
            sent_rank=np.argsort(-1 * pairwise_similarities[0]) # sort in descending order
            pred_answers.append(sent_list[sent_rank[0]])

    elif args.heuristic_type == 'bm25':
        for i in range(len(questions)):
            review_concat = ''
            for j in reviews[i]:
                review_concat += j + '' 
            sent_list = sent_tokenize(review_concat)
            tokenized_sent_list = [nltk.word_tokenize(sent) for sent in sent_list]
            bm25 = BM25Okapi(tokenized_sent_list)
            tokenized_query = nltk.word_tokenize(questions[i])
            top_sent = bm25.get_top_n(tokenized_query, tokenized_sent_list, n=1)
            #import IPython; IPython.embed(); exit(1)
            answer = TreebankWordDetokenizer().detokenize(top_sent[0])
            pred_answers.append(answer)

    elif args.heuristic_type == 'sentence_transformer':
        model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        for i in range(len(questions)):
            review_concat = ''
            for j in reviews[i]:
                review_concat += j + '' 
            sent_list = sent_tokenize(review_concat)
            sent_list.insert(0, questions[i])
            embeddings = model.encode(sent_list)
            pairwise_similarities=euclidean_distances(embeddings)
            sent_rank=np.argsort(-1 * pairwise_similarities[0]) # sort in descending order
            pred_answers.append(sent_list[sent_rank[0]])

    elif args.heuristic_type == 'doc2vec':
        for i in range(len(questions)):
            review_concat = ''
            for j in reviews[i]:
                review_concat += j + '' 
            sent_list = sent_tokenize(review_concat)
            sent_list.insert(0, questions[i])
            tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(sent_list)]

            model_d2v = Doc2Vec(vector_size=100,alpha=0.025, min_count=1) 
            model_d2v.build_vocab(tagged_data)

            for epoch in range(100):
                model_d2v.train(tagged_data,
                        total_examples=model_d2v.corpus_count,
                        epochs=model_d2v.epochs)
            
            document_embeddings=np.zeros((len(sent_list),100))

            for k in range(len(document_embeddings)):
                document_embeddings[k]=model_d2v.docvecs[k]

            pairwise_similarities=euclidean_distances(document_embeddings)
            sent_rank=np.argsort(-1 * pairwise_similarities[0]) # sort in descending order
            pred_answers.append(sent_list[sent_rank[0]])


    elif args.heuristic_type == 'glove':
        for i in range(len(questions)):
            review_concat = ''
            for j in reviews[i]:
                review_concat += j + '' 
            sent_list = sent_tokenize(review_concat)
            sent_list.insert(0, questions[i])

            tfidfvectoriser=TfidfVectorizer(max_features=100)
            tfidfvectoriser.fit(sent_list)
            tfidf_vectors=tfidfvectoriser.transform(sent_list)

            # tokenize and pad every document to make them of the same size
            tokenizer=Tokenizer()
            tokenizer.fit_on_texts(sent_list)
            tokenized_documents=tokenizer.texts_to_sequences(sent_list)
            tokenized_paded_documents=pad_sequences(tokenized_documents,maxlen=100,padding='post')
            vocab_size=len(tokenizer.word_index)+1

            # reading Glove word embeddings into a dictionary with "word" as key and values as word vectors
            embeddings_index = dict()

            with open('.vector_cache/glove.6B.100d.txt') as file:
                for line in file:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs

            # creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index. 
            embedding_matrix=np.zeros((vocab_size,100))

            for word,k in tokenizer.word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[k] = embedding_vector

            # tf-idf vectors do not keep the original sequence of words, converting them into actual word sequences from the documents
            document_embeddings=np.zeros((len(tokenized_paded_documents),100))
            words=tfidfvectoriser.get_feature_names()
            
            count = 0
            #import IPython; IPython.embed(); exit(1)
            for l in range(len(sent_list)):
                try:
                    for m in range(len(words)):
                        document_embeddings[l]+=embedding_matrix[tokenizer.word_index[words[m]]]*tfidf_vectors.toarray()[l][m].astype('float16')
                        count += 1
                        #print()
                        #print(count)
                except KeyError:
                    continue
        
            document_embeddings=document_embeddings/np.sum(tfidf_vectors,axis=1).reshape(-1,1)

            pairwise_similarities=euclidean_distances(document_embeddings)
            sent_rank=np.argsort(-1 * pairwise_similarities[0]) # sort in descending order
            pred_answers.append(sent_list[sent_rank[0]])


    output_json = {'Question':questions,
                    'Pred_answer':pred_answers,
                    'Ref_answers':multiple_answers}

    #df = pd.DataFrame(output_json, columns = ['Question', 'Pred_answer', 'Ref_answers'])
    #df.to_json(args.output_path, orient='split', index = False)

    with open(args.output_path, 'w') as json_file:
        json.dump(output_json, json_file)


if __name__ == "__main__":
    main()
