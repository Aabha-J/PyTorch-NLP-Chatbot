import nltk
#nltk.download('punkt')  # uncomment this if it's not downloaded, comment it out again after downloading
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(text):
    return nltk.word_tokenize(text)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_words, all_words):
    pass

if __name__ == "__main__":
    x = "Hello, I am a robot programmed to destroy"
    print(x)
    print(tokenize(x))

    test_words = ["organize", "organizes", "organizing"]
    stemmed_words = [stem(word) for word in test_words]
    print(stemmed_words)