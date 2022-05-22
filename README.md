# Neural Machine Translation
Machine translation refers to the translation of text from one language to another using computers.
Neural Machine translation is simply machine translation via neural networks. 
Neural machine translation is an approach to machine translation that uses an artificial neural network to predict the likelihood of a sequence of words, typically modeling entire sentences in a single integrated model.

# Algorithm
The model we will be building uses an encoder-decoder architecture. This 
Recurrent Neural Network (RNN) will take in a tokenized version of a sentence 
in its encoder, then passes it on to the decoder for translation. <br> <br>
Just using a regular sequence-to-sequence model with LSTMs will work 
effectively for short to medium sentences but will start to degrade for longer 
ones. Adding an attention layer to this model avoids this problem by giving the 
decoder access to all parts of the input sentence. <br> <br>
To produce the next prediction, the attention layer will first receive all the 
encoder hidden states as well as the decoder hidden state when producing the 
current word. Given this information, it will score each of the encoder hidden 
states to know which one the decoder should focus on to produce the next word. <br><br>

![image](https://user-images.githubusercontent.com/57310026/169679737-5ef8d75d-bb4a-4301-a3ee-cad90c27e51c.png)
![image](https://user-images.githubusercontent.com/57310026/169679739-144d3b26-0068-44dc-a68d-b0356d3e2cd2.png)

