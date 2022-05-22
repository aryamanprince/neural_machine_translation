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

## i)Steps
## Step1: Tokenization and Formatting
We want to represent each sentence as an array of integers instead of 
strings. We will assign a token (i.e. in this case 1) to mark the end of a 
sentence.
## Step2: Bucketing
Bucketing the tokenized sentences is an important technique used to speed up 
training in NLP. Our inputs have variable lengths and you want to make these 
the same when batching groups of sentences together. One way to do that is to 
pad each sentence to the length of the longest sentence in the dataset. This 
might lead to some wasted computation though. For example, if there are 
multiple short sentences with just two tokens, we don’t want to pad these when 
the longest sentence is composed of 100 tokens. Instead of padding with 0s to 
the maximum length of a sentence each time, we can group our tokenized 
sentences by length and bucket.
## Step3: Input Encoder
The input encoder runs on the input tokens, creates its embeddings, and feeds it 
to an LSTM network. This outputs the activations that will be the keys and 
values for attention. It is a Serial network which uses:
### • tl.Embedding: Converts each token to its vector representation. In this 
case, it is the the size of the vocabulary by the dimension of the 
model: tl.Embedding(vocab_size, d_model). vocab_size is the number of 
entries in the given vocabulary. d_model is the number of elements in the 
word embedding.
### • tl.LSTM: LSTM layer of size d_model. We want to be able to configure 
how many encoder layers we have so remember to create LSTM layers 
equal to the number of the n_encoder_layers parameter.
## Step4: Pre Attention Decoder
The pre-attention decoder runs on the targets and creates activations that are 
used as queries in attention. This is a Serial network which is composed of the 
following:
### • tl.ShiftRight: This pads a token to the beginning of your target tokens 
(e.g. [8, 34, 12] shifted right is [0, 8, 34, 12]). This will act like a start-ofsentence token that will be the first input to the decoder. During training, 
this shift also allows the target tokens to be passed as input to do teacher 
forcing.
### • tl.Embedding: Like in the previous function, this converts each token to 
its vector representation. In this case, it is the the size of the vocabulary 
by the dimension of the model: tl.Embedding(vocab_size, 
d_model). vocab_size is the number of entries in the given 
vocabulary. d_model is the number of elements in the word embedding.
### • tl.LSTM: LSTM layer of size d_model.
## Step5: Training
We will now be training our model in this section. Doing supervised training in 
Trax is pretty straightforward. We will be instantiating three classes for 
this: TrainTask, EvalTask, and Loop.
## Step6: Decoding
There are several ways to get the next token when translating a sentence. For 
instance, we can just get the most probable token at each step (i.e. greedy 
decoding) or get a sample from a distribution. We can generalize the 
implementation of these two approaches by using 
the tl.logsoftmax_sample() method.


We will import the dataset we will use to train the model. In the code, we are 
using English to German translation dataset form Opus/medical which contains 
medical related texts. Anyone can download dataset from https://opus.nlpl.eu/
