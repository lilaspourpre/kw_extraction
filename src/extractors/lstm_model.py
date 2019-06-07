import tensorflow as tf
from extractors.model import Model


class LSTMModel(Model):
    def __init__(self, iterator_tensors, hidden_size, learning_rate=0.0001):
        (lstm_input, input_length, expected_output, output_length) = iterator_tensors
        self.__encoder_lstm_size = hidden_size
        self.input = tf.identity(lstm_input, 'encoder_input')
        self.input_length = tf.identity(input_length, 'encoder_input_length')
        self.output = expected_output
        self.output_length = tf.identity(output_length, 'decoder_input_length')
        self.batch_size = tf.shape(self.input)[0]
        self.keep_prob = tf.placeholder_with_default(0.7, [], "keep_prob")
        self.outputs = self.get_output()
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.output))
        self.train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cross_entropy)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def get_output(self):
        transposed_input = tf.transpose(self.input, [1, 0, 2])
        with tf.variable_scope("fw"):
            cell = tf.contrib.rnn.LSTMBlockFusedCell(self.__encoder_lstm_size)
        fw_output, forward_encoder_state = cell(transposed_input, dtype=tf.float32,
                                                sequence_length=self.input_length)
        with tf.variable_scope("bw"):
            cell = tf.contrib.rnn.LSTMBlockFusedCell(self.__encoder_lstm_size)
        reversed_inputs = tf.reverse_sequence(transposed_input, self.input_length, seq_axis=0, batch_axis=1)
        bw_output, backward_encoder_state = cell(reversed_inputs, dtype=tf.float32,
                                                 sequence_length=self.input_length)
        bw_output = tf.reverse_sequence(bw_output, self.input_length, seq_axis=0, batch_axis=1)
        output = tf.concat([fw_output, bw_output], axis=-1)
        transposed_outputs = tf.transpose(output, [1, 0, 2])
        encoder_outputs = tf.nn.dropout(transposed_outputs, self.keep_prob)
        outputs = tf.contrib.layers.fully_connected(encoder_outputs, self.output_length, activation_fn=None,
                                                         scope="fc")
        mask = tf.expand_dims(tf.sequence_mask(self.output_length, dtype=tf.float32), -1)
        return outputs * mask


    def train(self, tagged_vectors, division):


    def predict(self, dataset, top_n):
        pass

    def save_to_path(self, path):
        pass
