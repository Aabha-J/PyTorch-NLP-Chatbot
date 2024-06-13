import random
import json
import torch

from model import NeuralNet
from data_processing import tokenize, bag_of_words


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_model(file_name):
  
  with open("intents.json", "r") as f:
      intents = json.load(f)

  data  = torch.load(file_name)

  input_size = data["input_size"]
  hidden_size = data["hidden_size"]
  output_size = data["output_size"]
  all_words = data["all_words"]
  tags = data["tags"]
  model_state = data["model_state"]

  model = NeuralNet(input_size, hidden_size, output_size).to(device)
  model.load_state_dict(model_state)
  model.eval()
  return model, all_words, tags, intents


def get_response(model, all_words, tags, sentence, intents):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)

    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.8:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand. Can you please try another question or rephrase your question"


if __name__ == "__main__":
    model, all_words, tags, intents = load_model("data.pth")
    bot_name = "Axel"

    print(f"Hi I'm {bot_name} Let's chat! (type 'quit' to exit)")

    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        get_response(model, all_words, tags, sentence, intents)
        print(f"{bot_name} : {get_response(model, all_words, tags, sentence, intents)}")