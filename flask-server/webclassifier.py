import pickle
from copy import deepcopy
import numpy as np
import json
import requests

# Declaring the global variables

input_path = "../bin/"

word_to_index = {}
index_to_word = {}

#the variables for the model input shape 
max_website_len = 1500
max_description_len = 100
max_keywords_len = 20

# The functions we need

def tokenize_questions(questions):
    noise_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you", "'re", "you", "ve", "you",
                   "'ll", "ll", "you", "'d", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                   'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
                   'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
                   'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do','does',
                   'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                   'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before','after',
                   'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                   'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both','each',
                   'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so','than',
                   'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd','ll',
                   'm', 'o', 're', 've', 'y', 'ain', 'aren', "are", 'couldn', "could", 'didn', "did", 'doesn', "does",
                   'hadn', "had", 'hasn', "has", 'haven', "have", 'isn', "is", 'ma', 'mightn', "might", 'mustn', "must",
                   "n't", 'needn', "need", 'shan', "sha", 'shouldn', "should", 'wasn', "was", 'weren', "were", 'won',"wo",
                   'wouldn', "would"]

    punctuation = '!"#$%&()*+,-./:;<=>@′[\]^_`”{|\}…“~ '
    tokenized = []

    for i, question in enumerate(questions):
        if (i % 10 == 0):
            print("{}/{}".format(i, len(questions)), end='\r', flush=True)

        for char in punctuation:
            question = question.replace(char, " ")
        for gram in ["n't", "'ll", "'re", "'ve", "n’t", "’ll", "’re", "’ve", "'s", "’s", "'m", "’m"]:
            question = question.replace(gram, " " + gram)

        tokens = question.split(' ')

        tokens = [token for token in tokens if len(token) > 0 and token != ' ' and token not in noise_words]

        tokenized.append(tokens)

    return tokenized


def index_to_words(question_indexes):
    question_words = ""

    for word in question_indexes:
        question_words += index_to_word[word] + ' '
    return question_words


def token_to_index(questions_tokenized):
    tokens = deepcopy(questions_tokenized)  # so the trasnformation is not in place

    for question in tokens:  # chage the input from strings to ids
        for idx, word_token in enumerate(question):  # replace the text with the word indexes
            question[idx] = word_to_index.get(word_token,
                                              word_to_index['unk'])  # if the token is not in vocab put "UNK"
    return tokens


def pad_questions(data, max_len=60):
    print("Padding...", end="")
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

        if len(data_aux[idx]) > max_len:
            break

    print("Done!")
    print("Total:", len(data), " Padded:", padded_count, " Cut:", cut_count )
    return np.vstack(data_aux)


def process(key_words, description, content):
    
    global word_to_index
    global index_to_word

    #get the global dictonaries so the functions can make the preprocesing
    with open(input_path + "word_to_index.h5", "rb" ) as file:
        word_to_index = pickle.load( file )

    with open(input_path + "index_to_word.h5", "rb" ) as file:
        index_to_word = pickle.load( file )

    #separate the text by word tokens
    key_words_input = tokenize_questions(np.array([key_words]))
    description_input = tokenize_questions(np.array([description]))
    content_input = tokenize_questions(np.array([content]))

    #change the word token to ints that mark the position of the word in the embedding layer
    #pad the input if it's too shord or cut it if it is too slong so it has the models inputs lenght 
    key_words_input = token_to_index(key_words_input)
    key_words_input = pad_questions(key_words_input, max_keywords_len)


    description_input = token_to_index(description_input)
    description_input = pad_questions(description_input, max_description_len)

    content_input = token_to_index(content_input)
    content_input = pad_questions(content_input, max_website_len)


    return key_words_input, description_input, content_input

def predict (key_words_input, description_input, content_input):
    payload = {
        "instances": [{'input_1:0':  content_input[0].tolist(),
                       'input_2:0':  key_words_input[0].tolist(),
                       'input_3:0':  description_input[0].tolist()
                     }]
    }

    r = requests.post('http://localhost:8000/v1/models/Classifier:predict', json=payload)

    pred = json.loads(r.content.decode('utf-8'))

    return pred['predictions'][0]