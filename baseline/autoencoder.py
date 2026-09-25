import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

class AutoEncoderNN:
    def __init__(self, n_input, n_classes, n_hidden_1=256, n_hidden_2=128, n_hidden_3=64):
        self.n_input = n_input
        self.n_classes = n_classes
        
        # --------------------- Encoder Variables --------------- #
        self.e_weights_h1 = weight_variable([n_input, n_hidden_1])
        self.e_biases_h1 = bias_variable([n_hidden_1])
        
        self.e_weights_h2 = weight_variable([n_hidden_1, n_hidden_2])
        self.e_biases_h2 = bias_variable([n_hidden_2])
        
        self.e_weights_h3 = weight_variable([n_hidden_2, n_hidden_3])
        self.e_biases_h3 = bias_variable([n_hidden_3])
        
        # --------------------- Decoder Variables --------------- #
        self.d_weights_h1 = weight_variable([n_hidden_3, n_hidden_2])
        self.d_biases_h1 = bias_variable([n_hidden_2])
        
        self.d_weights_h2 = weight_variable([n_hidden_2, n_hidden_1])
        self.d_biases_h2 = bias_variable([n_hidden_1])
        
        self.d_weights_h3 = weight_variable([n_hidden_1, n_input])
        self.d_biases_h3 = bias_variable([n_input])
        
        # --------------------- DNN Variables ------------------ #
        self.dnn_weights_h1 = weight_variable([n_hidden_3, n_hidden_2])
        self.dnn_biases_h1 = bias_variable([n_hidden_2])
        
        self.dnn_weights_h2 = weight_variable([n_hidden_2, n_hidden_2])
        self.dnn_biases_h2 = bias_variable([n_hidden_2])
        
        self.dnn_weights_out = weight_variable([n_hidden_2, n_classes])
        self.dnn_biases_out = bias_variable([n_classes])

    def encode(self, x):
        # Original paper logic: using tanh activation
        l1 = tf.nn.tanh(tf.add(tf.matmul(x, self.e_weights_h1), self.e_biases_h1))
        l2 = tf.nn.tanh(tf.add(tf.matmul(l1, self.e_weights_h2), self.e_biases_h2))
        l3 = tf.nn.tanh(tf.add(tf.matmul(l2, self.e_weights_h3), self.e_biases_h3))
        return l3
        
    def decode(self, x):
        # Original paper logic: using tanh activation
        l1 = tf.nn.tanh(tf.add(tf.matmul(x, self.d_weights_h1), self.d_biases_h1))
        l2 = tf.nn.tanh(tf.add(tf.matmul(l1, self.d_weights_h2), self.d_biases_h2))
        l3 = tf.nn.tanh(tf.add(tf.matmul(l2, self.d_weights_h3), self.d_biases_h3))
        return l3

    def dnn(self, x):
        # Original paper logic: using tanh activation, no dropout
        l1 = tf.nn.tanh(tf.add(tf.matmul(x, self.dnn_weights_h1), self.dnn_biases_h1))
        l2 = tf.nn.tanh(tf.add(tf.matmul(l1, self.dnn_weights_h2), self.dnn_biases_h2))
        out = tf.nn.softmax(tf.add(tf.matmul(l2, self.dnn_weights_out), self.dnn_biases_out))
        return out
