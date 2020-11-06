import tensorflow as tf
import config as cf

class Classify_Model():
    def __init__(
            self,
            learning_rate,
            vocab_size,
            embedding_size,
    ):
        self.lr = learning_rate
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.link_embeddings = tf.get_variable("link_embeddings", [vocab_size, embedding_size],
                                               initializer=tf.contrib.layers.xavier_initializer())
        self._build_network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_network(self):
        with tf.name_scope("input"):
            self.x_traj = tf.placeholder(tf.int32, [None, None], name="traj")
            self.x_traj_len = tf.placeholder(tf.int32, [None], name="traj_len")
            self.x_cross = tf.placeholder(tf.int32, [None, None], name="cross")
            self.x_start = tf.placeholder(tf.int32, [None], name="start")
            self.x_end = tf.placeholder(tf.int32, [None], name="end")
            self.state_c = tf.placeholder(tf.float32, [None, cf.lstm_hidden], name="state_c")
            self.state_h = tf.placeholder(tf.float32, [None, cf.lstm_hidden], name="state_h")
            self.y = tf.placeholder(tf.int32,[None, None], name="label")

        batch_size = tf.shape(self.x_traj)[0]
        max_len = tf.shape(self.x_traj)[1]

        nhidden = cf.lstm_hidden
        lstm_input = tf.nn.embedding_lookup(self.link_embeddings, self.x_traj)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(nhidden, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 1)
        #zero_state = lstm_cell.zero_state(batch_size, tf.float32)
        initial_state = tuple([tf.contrib.rnn.LSTMStateTuple(self.state_c, self.state_h)] * 1)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, lstm_input, dtype=tf.float32, initial_state = initial_state, sequence_length = self.x_traj_len)
        # outputs don't need [-1] shape = [batch_size, link_num, nhidden]
        self.out_state_c = states[-1][0] # [batch_size, nhidden]
        self.out_state_h = states[-1][1] # [batch_size, nhidden]

        start_embedding = tf.nn.embedding_lookup(self.link_embeddings, self.x_start) # [batch_size, embedding_size]
        start_embedding_tile = tf.tile(tf.expand_dims(start_embedding, axis=1), [1, max_len, 1]) # [batch_size, link_num, embedding_size]

        end_embedding = tf.nn.embedding_lookup(self.link_embeddings, self.x_end) # [batch_size, embedding_size]
        end_embedding_tile = tf.tile(tf.expand_dims(end_embedding, axis=1), [1, max_len, 1]) # [batch_size, link_num, embedding_size]

        cross_embedding = tf.nn.embedding_lookup(self.link_embeddings, self.x_cross) # [batch_size, link_num, embedding_size]

        inputs = tf.concat([start_embedding_tile, end_embedding_tile, outputs, cross_embedding], axis=2)

        inputs = tf.layers.dense(inputs, 512, activation=tf.nn.relu)
        inputs = tf.layers.dense(inputs, 512, activation=tf.nn.relu)
        inputs = tf.layers.dense(inputs, 256, activation=tf.nn.relu)
        inputs = tf.layers.dense(inputs, 128, activation=tf.nn.relu)

        self.output = tf.layers.dense(inputs, 2)
        
        self.prob = tf.nn.softmax(self.output)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.y)
        zero_like = tf.zeros_like(cross_entropy)
        cross_entropy = tf.where(tf.equal(self.x_cross, cf.MISS), zero_like, cross_entropy)
        self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        return self.loss

    def train(self, x_traj, x_traj_len, x_cross, x_start, x_end, state_c, state_h, y_batch):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.x_traj : x_traj,
            self.x_traj_len : x_traj_len,
            self.x_cross : x_cross,
            self.x_start: x_start,
            self.x_end: x_end,
            self.state_c: state_c,
            self.state_h: state_h,
            self.y : y_batch,
        })
        return loss

    def get_prob(self, x_traj, x_traj_len, x_cross, x_start, x_end, state_c, state_h):
        prob, out_state_c, out_state_h = self.sess.run([self.prob, self.out_state_c, self.out_state_h], feed_dict={
            self.x_traj : x_traj,
            self.x_traj_len : x_traj_len,
            self.x_cross : x_cross,
            self.x_start: x_start,
            self.x_end: x_end,
            self.state_c: state_c,
            self.state_h: state_h
        })
        return prob, out_state_c, out_state_h

    def save_model(self, save_path, global_steps):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path, global_steps)

    def link_embed(self):
        emb = self.sess.run(self.link_embeddings)
        print (emb[122565:122570,:4])

