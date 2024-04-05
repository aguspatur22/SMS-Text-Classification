# Spam Detection Model
This project is a machine learning model for classifying text messages as either "spam" or "ham" (not spam). The model is built using Keras, a popular deep learning library in Python.

## Model Architecture
The model architecture includes the following layers:
1. An Embedding layer that turns positive integers (indexes) into dense vectors of fixed size.
2. Dropout layers for regularization, set to drop out 20% of the nodes.
3. Two LSTM (Long Short-Term Memory) layers with 32 units each. LSTM is a type of recurrent neural network (RNN) that can learn and remember over long sequences and is not sensitive to sequence length.
4. A Batch Normalization layer, which normalizes the activations of the previous layer at each batch.
5. Two Dense layers, the last one being the output layer. The output layer uses a sigmoid activation function to output a probability that the text message is spam.

The model is compiled with the RMSprop optimizer and binary cross-entropy loss function, suitable for binary classification problems.

## Training
The model is trained on a balanced dataset for 16 epochs. The dataset is balanced by upsampling the minority class ("spam") to match the number of "ham" messages.

## Prediction
The predict_message function is used to make predictions on new text messages. The function preprocesses the input text, uses the model to predict the label, and returns a list containing the prediction and label. The prediction is a number between 0 and 1, and it is classified as 'ham' if it's less than 0.4 and 'spam' otherwise.
