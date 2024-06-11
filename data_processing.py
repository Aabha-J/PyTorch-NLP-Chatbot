import nltk
import json
import numpy as np
#nltk.download('punkt')  # uncomment this if it's not downloaded, comment it out again after downloading
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(text):
    return nltk.word_tokenize(text)

def stem(word):
    # Lowercases the word and stemming
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_words, all_words):
    """
    tokenized_words: ["hello", how", "are", "you"]
    all_words: ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    returns: [0, 1, 0, 1, 0, 0, 0]
    """
    tokenized_words = [stem(w) for w in tokenized_words]
    bag = np.zeros(len(all_words), dtype=np.float32) #[0, 0, 0, 0, 0, 0, 0]
    for idx, w in enumerate(all_words):
        if w in tokenized_words:
            bag[idx] = 1.0        #Flips the them to 1
    return bag

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

    return all_words, tags, xy

if __name__ == "__main__":
    #Testing Tokenization
    print("\nTesting Tokenization")
    print(tokenize("How are you doing?"))

    #Testing Stemming
    print("\nTesting Stemming")
    print(stem("organic"))

    #Testing get_json_data
    print("\nTesting get_json_data")
    all_words, tags, xy = get_json_data()
    print(all_words)
    print(tags)
    print(xy)


    #Testing Bag of Words
    print("\nTesting Bag of Words")
    tokenized_words =  ["hello", "how", "are", "you"]
    all_words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    print(bag_of_words(tokenized_words, all_words))
