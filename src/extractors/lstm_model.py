import tensorflow as tf
from extractors.model import Model


class LSTMModel(Model):
    def __init__(self, hidden_size, learning_rate=0.0001):
        #(lstm_input, input_length, expected_output) = iterator_tensors
        self.__encoder_lstm_size = hidden_size
        #self.n_class = 2
        #self.output = tf.one_hot(expected_output, depth = self.n_class)
        #self.input = tf.identity(lstm_input, name="input")
        #self.seq_length = tf.identity(input_length, name="seqlen")
        self.input = tf.placeholder(tf.float32, [None, None, 100], name='x')
        self.seq_length = tf.placeholder(tf.int32, [None], name='encoder_input_length')
        self.output = tf.placeholder(tf.float32, [None, None, 2], name='y')

        self.batch_size = tf.shape(self.input)[0]
        self.keep_prob = tf.placeholder_with_default(0.7, [], "keep_prob")
        self.outputs = self.get_output()
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.output))
        self.loss = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cross_entropy)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def get_output(self):
        transposed_input = tf.transpose(self.input, [1, 0, 2])
        with tf.variable_scope("fw"):
            cell = tf.contrib.rnn.LSTMBlockFusedCell(self.__encoder_lstm_size)
        fw_output, forward_encoder_state = cell(transposed_input, dtype=tf.float32,
                                                sequence_length=self.seq_length)
        with tf.variable_scope("bw"):
            cell = tf.contrib.rnn.LSTMBlockFusedCell(self.__encoder_lstm_size)
        reversed_inputs = tf.reverse_sequence(transposed_input, self.seq_length, seq_axis=0, batch_axis=1)
        bw_output, backward_encoder_state = cell(reversed_inputs, dtype=tf.float32,
                                                 sequence_length=self.seq_length)
        bw_output = tf.reverse_sequence(bw_output, self.seq_length, seq_axis=0, batch_axis=1)
        output = tf.concat([fw_output, bw_output], axis=-1)
        transposed_outputs = tf.transpose(output, [1, 0, 2])
        encoder_outputs = tf.nn.dropout(transposed_outputs, self.keep_prob)
        outputs = tf.contrib.layers.fully_connected(encoder_outputs, 2, activation_fn=None,
                                                         scope="fc")
        mask = tf.expand_dims(tf.sequence_mask(self.seq_length, dtype=tf.float32), -1)
        return outputs * mask

    def train(self, iterator):
        x, y = iterator.get_next()
        with self.sess as session:
            for i in range(10):
                session.run(iterator.initializer, feed_dict={self.input: x, self.output: y})
                try:
                    while True:
                        loss = session.run([self.loss])
                        print(loss)
                except tf.errors.OutOfRangeError:
                    print("end of epoch,", i)

    def predict(self, dataset, top_n):
        pass

    def save_to_path(self, path):
        pass
