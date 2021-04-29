import tensorflow as tf

#tf.compat.v1.disable_eager_execution()
import numpy as np
import past_queries
np.random.seed(1001)
tf.compat.v1.set_random_seed(1001)

## Model
class SentimentModel(object):
    def __init__(self, batch_size = 64, vocab_size=10000, max_len=200, lstm_size=64,
                 embeddings_dim=50, keep_probs=0.9, is_train=True):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.lstm_size = lstm_size
        self.keep_probs = keep_probs
        self.embeddings_dim = embeddings_dim
        self.is_train = is_train
        self.build_model()
            
    
    def build_model(self):
        # shape = (batch_size, sentence_length, word_id)
        self.x_holder = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_len])
        self.y_holder = tf.compat.v1.placeholder(tf.int64, shape=[None])
        self.seq_len = tf.cast(tf.reduce_sum(tf.sign(self.x_holder), axis=1), tf.int32)
        with tf.device("/cpu:0"):
            # embeddings matrix
            self.embedding_w = tf.compat.v1.get_variable('embed_w', shape=[self.vocab_size,self.embeddings_dim],
                                           initializer=tf.random_uniform_initializer(), trainable=True)
            # embedded words
            self.e = tf.nn.embedding_lookup(self.embedding_w, self.x_holder)
        
        
        lstm = tf.compat.v1.nn.rnn_cell.LSTMCell(self.lstm_size)
        #lstm = tf.keras.layers.GaussianNoise(.005)
        #lstm = tf.keras.layers.LSTMCell(self.lstm_size)(lstm)
        if self.is_train:
            #lstm = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm, state_keep_prob=0.5)
            self.e = tf.nn.dropout(self.e, self.keep_probs)
        self.init_state = lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        #self.init_state= lstm.get_initial_state(batch_size=self.batch_size, dtype=tf.float32) 
        
        rnn_outputs, final_state = tf.compat.v1.nn.dynamic_rnn(cell=lstm,
                                                    inputs=self.e,
                                                    initial_state=self.init_state,
                                                    sequence_length=self.seq_len)
        batch_size = tf.shape(rnn_outputs)[0]
        max_length = tf.shape(rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (self.seq_len - 1)
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        relevant = tf.gather(flat, index)
        last_output = rnn_outputs[:,-1,:]
        relevant = tf.reduce_mean(rnn_outputs, axis=1)
        last_output = tf.nn.dropout(last_output, 0.25)
        last_output = relevant
        if self.is_train:
            last_output = tf.nn.dropout(last_output, self.keep_probs)
        self.w = tf.compat.v1.get_variable("w", shape=[self.lstm_size, 2], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.2))
        self.b = tf.compat.v1.get_variable("b", shape=[2], dtype=tf.float32)
        logits = tf.matmul(last_output, self.w) + self.b
        self.y = tf.nn.softmax(logits)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.y_holder, depth=2),logits=logits))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_holder, tf.argmax(self.y, 1)), tf.float32))
        
        if self.is_train:
            #print(self.cost)
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0)
            self.train_op = self.optimizer.minimize(self.cost)
        
    def train_for_epoch(self, sess, train_x, train_y):
        #cur_state = sess.run(init_state)
        assert self.is_train, 'Not training model'
        batches_per_epoch = train_x.shape[0] // self.batch_size
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for idx in range(batches_per_epoch):
            batch_idx = np.random.choice(train_x.shape[0], size=self.batch_size, replace=False)
            batch_xs = train_x[batch_idx,:]
            batch_ys = train_y[batch_idx]
            batch_loss, _, batch_accuracy = sess.run([self.cost, self.train_op, self.accuracy],
                                     feed_dict={self.x_holder: batch_xs,
                                               self.y_holder: batch_ys})
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
        return epoch_loss / batches_per_epoch, epoch_accuracy / batches_per_epoch
            #print(batch_xs.shape)

    def fill_past_queries(self, current_query, current_output):
        # if we reached the end of the past_queries momery
        if past_queries.index + 1 > past_queries.queries.shape[0]:
            past_queries.queries[0] = current_query
            past_queries.outputs[0] = current_output
        
        else:
            past_queries.queries[past_queries.index] = current_query
            past_queries.outputs[past_queries.index] = current_output

        past_queries.index = (past_queries.index + 1) % past_queries.queries.shape[0]
        return

    def attack_detection(self, sess,test_x, 
                                normalizer_range=25,
                                threshold = 0.95,
                                threshold2 = 0.8999999,
                                thershold2_min_queries = 5):        
        embedding_x = tf.nn.embedding_lookup(self.embedding_w, test_x) #Shape : (1,250,300)
        embedding_sentence = tf.math.reduce_mean(embedding_x, 1) #Shape :(1,300)

        min = tf.math.reduce_min(embedding_sentence, keepdims=True,axis=-1)
        max = tf.math.reduce_max(embedding_sentence, keepdims=True, axis=-1)
        embedding_sentence_normalize = normalizer_range * (embedding_sentence-min)/(max-min)
        
        #round
        embedding_sentence_normalize_round = tf.cast(tf.math.round(embedding_sentence_normalize), dtype=tf.uint8)

        #convert to numpy
        embedding_sentence_normalize_round =embedding_sentence_normalize_round.eval(session=sess)

        #ANALYZING THE QUERIES
        is_same = None
        pred_y = sess.run(self.y, feed_dict={self.x_holder: test_x})
        past_queries.number_pred += test_x.shape[0]

        for i, output in zip(range(test_x.shape[0]), pred_y):
            
            is_same = (past_queries.queries == embedding_sentence_normalize_round[i])   
            match_ratio = is_same.sum(-1)/is_same.shape[-1]
                
            similar_queries = match_ratio > threshold
            similar_queries2 = match_ratio > threshold2
            # There is an attack if there is some queries that have an match_ratio higher than threshold
            if similar_queries.any(): 
                similar_queries_output = past_queries.outputs[similar_queries]

                majoritary_class = int(np.mean(similar_queries_output.argmax(axis=-1)).round())
                max_pred_index = np.argmax(similar_queries_output[:, majoritary_class])               
                pred_y[i] = similar_queries_output[max_pred_index]                
                
                #MEAN
                #pred_y[i] = np.mean(similar_queries_output, axis=0)  

                past_queries.number_attack_detected += 1

            elif similar_queries2.sum() >= thershold2_min_queries:
                #TODO : put this in a method 
                similar_queries_output = past_queries.outputs[similar_queries2]
                
                majoritary_class = int(np.mean(similar_queries_output.argmax(axis=-1)).round())
                max_pred_index = np.argmax(similar_queries_output[:, majoritary_class])               
                
                # Update pred_y
                pred_y[i] = similar_queries_output[max_pred_index]
                
                #MEAN
                #pred_y[i] = np.mean(similar_queries_output, axis=0) 

                past_queries.number_attack_detected += 1

            self.fill_past_queries(embedding_sentence_normalize_round[i], pred_y[i])

        return pred_y

    def predict(self, sess, test_x, defense_activated=True):
        if defense_activated: 
            pred_y = self.attack_detection(sess, test_x)
        else:
            past_queries.number_pred += test_x.shape[0]
            pred_y = sess.run(self.y, feed_dict={self.x_holder: test_x})
        return pred_y
    
    def evaluate_accuracy(self, sess, test_x, test_y):
        print(self.batch_size)
        test_accuracy = 0.0
        test_batches = test_x.shape[0] // self.batch_size
        for i in range(test_batches):
            test_idx = range(i*self.batch_size, (i+1)*self.batch_size)
            test_xs = test_x[test_idx,:]
            test_ys = test_y[test_idx]
            pred_ys = self.predict(sess, test_xs)
            test_accuracy += np.sum(np.argmax(pred_ys, axis=1) == test_ys)
        test_accuracy /= (test_batches*self.batch_size)
        return test_accuracy