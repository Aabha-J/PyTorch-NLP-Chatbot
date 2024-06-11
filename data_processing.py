import nltk
import json
#nltk.download('punkt')  # uncomment this if it's not downloaded, comment it out again after downloading
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(text):
    return nltk.word_tokenize(text)

def stem(word):
    # Lowercases the word and stemming
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_words, all_words):
    pass

def get_json_data():
    with open("intents.json", "r") as f:
        data = json.load(f)


    all_words = []
    tags = []
    xy = []
    for intent in data["intents"]:
        tag = intent["tag"]
        tags.append(tag)

        # 1. Tokenization
        for pattern in intent["patterns"]:
            w = tokenize(pattern)
            all_words.extend(w) # extend adds all the elements of w to all_words
            xy.append((w, tag))

    ignore_words = ["?", "!", ".", ","]


    #2. Lowercasing (done in stemming function) and 3. Stemming and 
    all_words = [stem(w) for w in all_words if w not in ignore_words] #4. Remove Punctuation
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # print(len(xy), "patterns")
    # print(len(tags), "tags:", tags)
    # print(len(all_words), "unique stemmed words:", all_words)

    return all_words, tag, xy
