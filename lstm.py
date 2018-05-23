# coding = utf-8
import tensorflow as tf


def build_rnn(hidden_units=128, layers=2, batch_size=1, vocab_size=0,input_sequences=None):
	with tf.variable_scope("embedding"):
		embedding = tf.get_variable("embedding", [vocab_size, hidden_units], dtype=tf.float32)
		# input: batch_size * time_step * embedding_feature
		input = tf.nn.embedding_lookup(embedding, input_sequences)

	# 定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
	basic_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units, state_is_tuple=True)
	# 调用 MultiRNNCell 来实现多层 LSTM
	stack_cell = tf.nn.rnn_cell.MultiRNNCell([basic_cell]*layers)
	# 用全零来初始化state
	_initial_state = stack_cell.zero_state(batch_size, tf.float32)

	outputs, state = tf.nn.dynamic_rnn(stack_cell, input, initial_state=_initial_state, dtype=tf.float32)
	outputs = tf.reshape(outputs, [-1, hidden_units])

	with tf.variable_scope("softmax"):
		softmax_w = tf.get_variable("softmax_w", [hidden_units, vocab_size])
		softmax_b = tf.get_variable("softmax_b", [vocab_size])
		logits = tf.matmul(outputs, softmax_w) + softmax_b

	probs = tf.nn.softmax(logits)

	return logits, probs, stack_cell, _initial_state, state