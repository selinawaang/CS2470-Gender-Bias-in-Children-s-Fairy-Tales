import tensorflow as tf
import numpy as np
from preprocess import get_data
from types import SimpleNamespace


class MyRNN(tf.keras.Model):

    ##########################################################################################

    def __init__(self, vocab_size, rnn_size=128, embed_size=64):
        """
        The Model class predicts the next words in a sequence.
        : param vocab_size : The number of unique words in the data
        : param rnn_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size

        ## TODO:
        ## - Define an embedding component to embed the word indices into a trainable embedding space.
        self.embed_matrix = tf.keras.layers.Embedding(self.vocab_size, self.embed_size)
        ## - Define a recurrent component to reason with the sequence of data. 
        self.lstm = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        ## - You may also want a dense layer near the end...
        self.dense = tf.keras.layers.Dense(self.vocab_size, activation='softmax')    

    def call(self, inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup or tf.keras.layers.Embedding)
        - You must use an LSTM or GRU as the next layer.
        """
        x = inputs
        x = self.embed_matrix(x)
        x, final_output, final_state = self.lstm(x)
        #print(x.shape)
        x = self.dense(x)



        #print(whole_seq_output.shape, final_output.shape, print(final_output))

        return x

    ##########################################################################################

    def generate_sentence(self, word1, length, vocab, sample_n=10):
        """
        Takes a model, vocab, selects from the most likely next word from the model's distribution.
        (NOTE: you shouldn't need to make any changes to this function).
        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}

        first_string = word1
        first_word_index = vocab[word1]
        next_input = np.array([[first_word_index]], dtype=np.int32)
        text = [first_string]

        for i in range(length):
            logits = self.call(next_input)
            #print(logits)
            logits = np.array(logits[0,0,:])
            top_n = np.argsort(logits)[-sample_n:]
            n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
            out_index = np.random.choice(top_n,p=n_logits)

            text.append(reverse_vocab[out_index])
            next_input = np.array([[out_index]], dtype=np.int32)

        print(" ".join(text))


#########################################################################################

def get_text_model(vocab):
    '''
    Tell our autograder how to train and test your model!
    '''

    ## TODO: Set up your implementation of the RNN

    ## Optional: Feel free to change or add more arguments!
    model = MyRNN(len(vocab))

    def perplexity(y_true, y_pred):
        return tf.math.exp(tf.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(y_true, y_pred)))


    # TODO: Define your own loss and metric for your optimizer
    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy()
    acc_metric  = perplexity
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    model.compile(
        optimizer=adam_optimizer, 
        loss=loss_metric, 
        metrics=[acc_metric],
    )

    return SimpleNamespace(
        model = model,
        epochs = 2,
        batch_size = 100,
    )



#########################################################################################

def main():

    ## TODO: Pre-process and vectorize the data
    ##   HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    ##   from train_x and test_x. You also need to drop the first element from train_y and test_y.
    ##   If you don't do this, you will see very, very small perplexities.
    ##   HINT: You might be able to find this somewhere...
    train_file = '../data/train.txt'
    test_file = '../data/test.txt'

    train_data, test_data, vocab = get_data(train_file, test_file)

    # offfset words
    X0, Y0  = train_data[:-1], train_data[1:]
    X1, Y1  = test_data[:-1], test_data[1:]

    train_len = X0.shape[0] - (X0.shape[0] % 20)
    test_len = X1.shape[0] - (X1.shape[0] % 20)


    X0 = X0[:train_len]
    Y0 = Y0[:train_len]
    X1 = X1[:test_len]
    Y1 = Y1[:test_len]

    X0 = tf.reshape(X0,[-1,20])
    Y0 = tf.reshape(Y0,[-1,20])
    X1 = tf.reshape(X1,[-1,20])
    Y1 = tf.reshape(Y1,[-1,20])

    #print(Y0.shape)

    print(X0.shape, Y0.shape, X1.shape, Y1.shape)


    # window_sz = 20
    # for i in range(window_sz,X0.shape[0],window_sz):
    # 	X_train.append(X0[i-window_sz:i])
    # 	y_train.append(Y0[i-window_sz:i])
    # for i in range(window_sz, X1.shape[0],window_sz):
    # 	X_test.append(X1[i-window_sz:i])
    # 	y_test.append(Y1[i-window_sz:i])
    ## TODO: Get your model that you'd like to use
    # X_train = np.array(X_train)
    # X_train = tf.constant(X_train)

    # y_train = np.array(y_train)
    # y_train = tf.constant(y_train)

    # X_test = np.array(X_test)
    # X_test = tf.constant(X_test)

    # y_test = np.array(y_test)
    # y_test = tf.constant(y_test)



    args = get_text_model(vocab)

    args.model.fit(
        X0, Y0,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_data=(X1, Y1)
    )

    ## Feel free to mess around with the word list to see the model try to generate sentences
    for word1 in 'speak to this brown deep learning student'.split():
        if word1 not in vocab: print(f"{word1} not in vocabulary")            
        else: args.model.generate_sentence(word1, 20, vocab, 10)

if __name__ == '__main__':
    main()
