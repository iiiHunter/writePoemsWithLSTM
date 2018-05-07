# coding = utf-8
import numpy as np
import tensorflow as tf
import os
from prepare_data import get_data
from lstm import build_rnn


def train(reload=False):
    model_save_path = os.getcwd() + "/peotry/peotry"
    # build rnn
    input_sequences = tf.placeholder(tf.int32, shape=[batch_size, None])
    output_sequences = tf.placeholder(tf.int32, shape=[batch_size, None])
    logits, probs, _, _, _ = build_rnn(batch_size=batch_size, vocab_size=vocab_size,input_sequences=input_sequences)
    targets = tf.reshape(output_sequences, [-1])

    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], len(words))
    cost = tf.reduce_mean(loss)

    learning_rate = tf.Variable(0.002, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    global_step = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

        if reload:
            module_file = tf.train.latest_checkpoint("./peotry")
            start_epoch = int(module_file.split('-')[-1])
            saver.restore(sess, module_file)
            print("reload sess from file successfully!")
        else:
            start_epoch = 0

        for epoch in range(start_epoch, 50):
            print("one more epoch, learning_rate decrease")
            if global_step % 80 == 0:
                sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
            epoch_steps = len(list(zip(X_data, Y_data)))
            for step, (x, y) in enumerate(zip(X_data, Y_data)):
                global_step = epoch * epoch_steps + step
                _, los = sess.run([train_op, cost], feed_dict={
                    input_sequences: x,
                    output_sequences: y,
                    })
                if global_step % 100 == 0:
                    print("epoch:%d steps:%d/%d loss:%3f" %(epoch, step, epoch_steps, los))
                if global_step % 1000 == 0:
                    print(" ======  save model in " + model_save_path + "  ====== ")
                    saver.save(sess, model_save_path, global_step=epoch)

if __name__ == '__main__':
    batch_size = 1
    X_data, Y_data, words, word2idfunc = get_data(poetry_file='data/poetry.txt', batch_size=batch_size)
    vocab_size = len(words) + 1
    # input_size:(batch_size, feature_length)
    train(reload=True)
