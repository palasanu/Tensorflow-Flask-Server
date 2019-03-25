import argparse
import json
import pickle as pickle
import pandas  as pd
import numpy as np
import requests
from copy import deepcopy

input_path = "../bin/"

max_website_len = 1500
max_description_len = 100
max_keywords_len = 20

def tokenize_questions(questions):
    
    noise_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you","'re", "you", "ve", "you",
                   "'ll", "ll", "you","'d", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                   'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 
                   'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
                   'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
                   'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                   'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                   'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under','again', 
                   'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
                   'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                   'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
                   'm', 'o', 're', 've', 'y', 'ain', 'aren', "are", 'couldn', "could", 'didn', "did", 'doesn', "does",
                   'hadn', "had", 'hasn', "has", 'haven', "have", 'isn', "is", 'ma', 'mightn', "might", 'mustn', "must", 
                   "n't", 'needn', "need", 'shan', "sha", 'shouldn', "should", 'wasn', "was", 'weren', "were", 'won', "wo",
                   'wouldn', "would"]
    
    punctuation = '!"#$%&()*+,-./:;<=>@′[\]^_`”{|\}…“~ '
    tokenized = []
   
    for i, question in enumerate(questions):
        if (i%10==0):
            print ("{}/{}".format(i , len(questions)), end = '\r', flush = True)
        for char in punctuation:
            question = question.replace(char, " ")
        for gram in ["n't", "'ll","'re","'ve", "n’t", "’ll", "’re", "’ve","'s", "’s", "'m" , "’m"]:
            question = question.replace(gram, " "+gram)

        tokens = question.split(' ')

        tokens = [token for token in tokens if len(token)>0 and token != ' ' and token not in noise_words]

        tokenized.append(tokens)

    return tokenized

def index_to_words(question_indexes):

    question_words = ""

    for word in question_indexes:
        question_words += index_to_word[word] + ' '
    return question_words

def token_to_index(questions_tokenized):
    
    tokens = deepcopy(questions_tokenized) #so the trasnformation is not in place
    
    for question in tokens: #chage the input from strings to ids
        for idx,word_token in enumerate(question): #replace the text with the word indexes
             question[idx] = word_to_index.get(word_token, word_to_index['unk']) #if the token is not in vocab put "UNK"
    return tokens


def pad_questions(data, max_len = 60):
    
    print("Padding...", end = "")
    data_aux = deepcopy(data)
    
    padded_count = 0
    cut_count = 0
    for idx in range(len(data_aux)):
        question_len = len(data_aux[idx])
        if question_len <= max_len:
            padded_count += 1
            for i in range(question_len, max_len):
                data_aux[idx].append(word_to_index["<pad>"])
        elif question_len > max_len:
            cut_count += 1
            data_aux[idx] = data_aux[idx][:max_len]
        
        if len(data_aux[idx])>max_len:
            break
    
    print("Done!")
    print("Total:", len(data), " Padded:", padded_count, " Cut:", cut_count )

    return np.vstack(data_aux)


with open(input_path + "word_to_index.h5", "rb" ) as file:
    word_to_index = pickle.load( file )

with open(input_path + "index_to_word.h5", "rb" ) as file:
    index_to_word = pickle.load( file )


data = pd.read_csv("../test_data/test_data.csv")[:10]


description = tokenize_questions( data["description"].values )
key_words = tokenize_questions( data["keyword"].values )
content = tokenize_questions( data["content"].values )

content = token_to_index(content)
content = pad_questions(content, max_website_len)

description = token_to_index(description)
description = pad_questions(description, max_description_len)

key_words = token_to_index(key_words)
key_words = pad_questions(key_words, max_keywords_len)


payload = {
    "instances": [  {'input_1:0': content[0].tolist(),
                     'input_2:0': key_words[0].tolist(),
                     'input_3:0': description[0].tolist()
                    }
    ]
}

r = requests.post('http://localhost:8000/v1/models/Classifier:predict', json=payload)

pred = json.loads(r.content.decode('utf-8'))

print(pred)