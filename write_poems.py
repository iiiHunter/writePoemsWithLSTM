# coding = utf-8
import numpy as np
import tensorflow as tf
import os
from prepare_data import get_data
from lstm import build_rnn

def write_poem():  # 根据概率分布选择与直接选择可能性最大的
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        sample2 = int(np.argmax(weights))
        print("sample:%d/%d:" % (sample, len(words)))
        print("sample2:%d/%d:" % (sample2, len(words)))
        print("==============")
        return words[sample], words[sample2]

    logits, probs, stack_cell, _initial_state, last_state = build_rnn(batch_size=batch_size,vocab_size=vocab_size)
    # input_size:(batch_size, feature_length)
    input_sequences = tf.placeholder(tf.int32, shape=[batch_size, None])
    with tf.Session() as sess:
        # sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        module_file = tf.train.latest_checkpoint("./peotry")
        print("load sess from file:", module_file)
        saver.restore(sess, module_file)

        _state = sess.run(stack_cell.zero_state(1, dtype=tf.float32))

        x = np.array([[word2idfunc('[')]])

        prob_, _state = sess.run([probs, last_state], feed_dict={input_sequences: x, _initial_state: _state})

        word, word2 = to_word(prob_)

        poem = ''
        poem2 = ''

        while word != ']':
            poem += word
            poem2 += word2
            x = np.array([[word2idfunc(word)]])
            [probs_, _state] = sess.run([probs, last_state], feed_dict={input_sequences: x, _initial_state: _state})
            word, word2 = to_word(probs_)

    return poem, poem2


def write_head_poem(heads):
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        print("sample:", sample)
        print("len Words:", len(words))
        # sample = np.argmax(weights)
        return words[sample]

    logits, probs, stack_cell, _initial_state, last_state = build_rnn()
    input_sequences = tf.placeholder(tf.int32, shape=[batch_size, None])

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        module_file = tf.train.latest_checkpoint('./peotry')
        print("load ress from file:", module_file)
        saver.restore(sess, module_file)

        _state = sess.run(stack_cell.zero_state(1, dtype=tf.float32))

        poem = ''
        add_comma = False
        for head in heads:
            x = head
            add_comma = not add_comma
            while x != "," and x != "。" and x != ']':
                # add current
                poem += x
                x = np.array([[word2idfunc(x)]])
                # generate next based on current
                prob_, _state = sess.run([probs, last_state], feed_dict={input_sequences: x, _initial_state: _state})
                x = to_word(prob_)
            sign = ", " if add_comma else "。"
            poem = poem + sign
        return poem

if __name__ == '__main__':
    batch_size = 1
    _, _, words, word2idfunc = get_data(poetry_file='data/poetry.txt', batch_size=batch_size)
    vocab_size = len(words) + 1
    r1,r2 = write_poem()
    print(r1)
    print(r2)
    #print(write_head_poem(u"春风十里"))
