# Simple Clothes Store Chatbot 🤖
I used pytorch and nltk to train an ML model and create a chatbot that can tell some jokes and anwser simple questions 
about a typical clothes store. 

# Demo
![Untitled video - Made with Clipchamp](https://github.com/Aabha-J/PyTorch-NLP-Chatbot/assets/121515351/1854c93e-798c-4713-b724-b05265acf3b5)

# Running the Project
Follow the instructions or checkout the setup.txt file

  To install nesseacry libraries
    1. Set up a virtual python enviorment
    2.   pip install numpy streamlit torch nltk
  
  To run chat in terminal:
    python run chat.python
  
  To run chat with streamlit UI:
    streamlist run app.py

# Theory
  The theory behind the ML model. This info is also available in the Theory.txt file


## Data Processing
  Steps Taken to Process the Data
    1. Tokenization
    2. Put Everything to Lowercase
    3. Stemming
    4. Remove Punctuation
    5. Bag of Words

Tokenization: Splitting a string into meaningful units
 (ex:words, punctuation, numbers)

Stemming: Generates the root form of the words. Chops of the ends of the words
 Ex: organize, organizes, organizer all become organ

Bag of Words: Collects different words and puts it into an array. Occurence of each words
in the array is used by the model to map input to output

## The Model Used:
  Feed Forward Neural Net
      Input: Bag of Words
      Output: Tag


