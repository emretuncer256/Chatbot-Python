# Chatbot
This is a simple chatbot that uses a neural network to classify input text into different intents and generate responses based on predefined patterns. It has three main components: train.py, main.py, and intents.json.

- ## train.py
This script preprocesses the data, trains a neural network model using Keras, and saves the trained model and the training data to disk. The data is stored in a pickle file that contains the words, classes, training patterns, and output labels.

- ## main.py
This script loads the trained model and the training data, and uses them to classify input text and generate responses. The input text is first preprocessed using a bag-of-words approach and then passed through the neural network to generate the predicted intent. The script then selects a response from a predefined set of responses based on the predicted intent.

- ## intents.json
This file contains a set of predefined patterns and responses that the chatbot uses to classify input text into different intents and generate appropriate responses. The file is structured as a JSON object with two main keys:

"intents": a list of intent objects, each with a "tag" key representing the intent name and a "patterns" key representing a list of input text patterns that correspond to the intent.
"responses": a list of response objects, each with a "tag" key representing the intent name and a "responses" key representing a list of possible responses for the intent.
For example, here is a sample "intents" object from the file:
````json
    "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi there", "Hello", "Greetings", "What's up", "Hey"],
      "responses": ["Hello", "Hi", "Hi there"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later", "Goodbye"],
      "responses": ["See you later", "Have a nice day", "Goodbye!"]
    },
    ...
]
````
This indicates that the chatbot recognizes the "greeting" intent when the input text matches one of the patterns in the "patterns" list, and generates a response from the "responses" list. Similarly, it recognizes the "goodbye" intent and generates a response from the corresponding "responses" list.

To use the chatbot, simply run the main.py script and enter text input. The chatbot will respond with a relevant message based on its prediction. To exit the chatbot, type "quit".

Note that the chatbot's performance may be limited by the quality and quantity of the training data, as well as by the complexity of the neural network architecture. Further improvements can be made by adding more training data, fine-tuning the neural network, and adding more sophisticated natural language processing techniques.

## Screenshot
![Screenshot](screenshott/sc1.png)
## License
This project is licensed under the MIT License - see the LICENSE file for details