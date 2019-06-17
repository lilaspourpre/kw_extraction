import os
import tensorflow as tf
import time
from extractors.model import Model


class LSTMModel(Model):
    def __init__(self, iterator_tensors, hidden_size=512, output_size=1, learning_rate=0.0001,
                 batch_print_rate=10, model_path=None):
        (lstm_input, input_length, expected_output) = iterator_tensors
        self.__encoder_lstm_size = hidden_size
        self.__output_size = output_size
        self.batch_print_rate = batch_print_rate
        self.encoder_input = tf.identity(lstm_input, 'encoder_input')
        self.encoder_input_length = tf.identity(input_length, 'encoder_input_length')
        self.expected_output = tf.identity(expected_output, 'expected_output')
        self.output_length = tf.identity(input_length, 'output_length')
        self.batch_size = tf.shape(self.encoder_input)[0]
        self.keep_prob = tf.placeholder_with_default(0.7, [], "keep_prob")
        self.outputs = self.get_output()
        self.cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.outputs, labels=self.expected_output))
        self.loss = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cross_entropy)
        self.predicted = tf.round(tf.nn.sigmoid(self.outputs))
        self.correct_pred = tf.equal(self.predicted, self.expected_output)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.saver = tf.train.Saver()
        self.session = tf.Session(config=tf.ConfigProto())
        if model_path:
            self.saver.restore(self.session, model_path + '/variables/variables')
            print("Restored {}".format(model_path))
        else:
            self.init = tf.global_variables_initializer()
            self.session.run(self.init)

    def get_output(self):
        transposed_input = tf.transpose(self.encoder_input, [1, 0, 2])
        with tf.variable_scope("fw"):
            cell = tf.contrib.rnn.LSTMBlockFusedCell(self.__encoder_lstm_size)
        fw_output, forward_encoder_state = cell(transposed_input, dtype=tf.float32, sequence_length=self.encoder_input_length)
        with tf.variable_scope("bw"):
            cell = tf.contrib.rnn.LSTMBlockFusedCell(self.__encoder_lstm_size)
        reversed_inputs = tf.reverse_sequence(transposed_input, self.encoder_input_length, seq_axis=0, batch_axis=1)
        bw_output, backward_encoder_state = cell(reversed_inputs, dtype=tf.float32,
                                                 sequence_length=self.encoder_input_length)
        bw_output = tf.reverse_sequence(bw_output, self.encoder_input_length, seq_axis=0, batch_axis=1)
        output = tf.concat([fw_output, bw_output], axis=-1)
        transposed_outputs = tf.transpose(output, [1, 0, 2])
        encoder_outputs = tf.nn.dropout(transposed_outputs, self.keep_prob)
        outputs = tf.contrib.layers.fully_connected(encoder_outputs, self.__output_size, activation_fn=None,
                                                         scope="fc")
        mask = tf.expand_dims(tf.sequence_mask(self.output_length, dtype=tf.float32), -1)
        return outputs * mask

    def train(self, iterator, output_path, epochs=2):
        for epoch in range(epochs):
            print("[{}] Epoch {} started".format(time.strftime("%H:%M:%S"), str(epoch)))
            losses = []
            accuracies = []
            batch_counter = 0
            self.session.run(iterator.initializer)
            while True:
                try:
                    loss, _, accuracy = self.session.run([self.cross_entropy, self.loss, self.accuracy])
                    losses.append(loss)
                    accuracies.append(accuracy)
                    if batch_counter % self.batch_print_rate == 0:
                        print("[{}] Epoch: {}, loss: {}, avg_loss: {}, accuracy: {}, avg_accuracy: {}".format(time.strftime("%H:%M:%S"),
                                                                          epoch, loss, self.__avg(losses), accuracy, self.__avg(accuracies)))
                    batch_counter += 1
                except tf.errors.OutOfRangeError:
                    break
            self.save_to_path(os.path.join(output_path+str(epoch)+"epoch_{}".format(time.strftime('%Y.%m.%d-%H.%M.%S'))))

    def __avg(self, numbers):
        return sum(numbers) / len(numbers)

    def predict(self, iterator, top_n):
        self.session.run(iterator.initializer)
        predictions = []
        true_labels = []
        while True:
            try:
                predicted, output = self.session.run([self.predicted, self.expected_output])
                predictions.extend(predicted)
                true_labels.extend(output)
            except tf.errors.OutOfRangeError:
                break
        return predictions, true_labels

    def save_to_path(self, path):
        model_saver = tf.saved_model.builder.SavedModelBuilder(path)
        model_saver.add_meta_graph_and_variables(
            self.session,
            [tf.saved_model.tag_constants.SERVING])
        model_saver.save()
