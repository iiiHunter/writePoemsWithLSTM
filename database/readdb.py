#!/usr/bin/env python3
# coding: utf-8
import os
import json
import collections
import numpy as np


def read_db(file_path, author=None, tag=None):
    poetrys = []
    with open(file_path) as file:
        file_content = json.load(file)
        poet_num = len(file_content)
        with open("/home/iiihunter/writePoems/database/json/tags.json") as tags_file:
            tags = json.load(tags_file)
        for i in range(poet_num):
            one_poet = file_content[i]
            paragraph = one_poet["paragraphs"]
            _author = one_poet["author"]

            if tag != None:
                if tag not in tags:
                    print("Tag error, the tag is not exit!")
                if "tags" not in one_poet.keys() or (tag not in one_poet["tags"]):
                    continue
            if author != None:
                if _author != author:
                    continue

            poet_len = len(paragraph)
            one_poet_str = "["
            for i in range(poet_len):
                one_poet_str = one_poet_str + paragraph[i]
            one_poet_str = one_poet_str + "]"
            if '_' in one_poet_str or '（' in one_poet_str or '（' in one_poet_str \
                    or '《' in one_poet_str:
                continue
            poetrys.append(one_poet_str)
    return poetrys, tags


def get_data(poetry_file=None, batch_size=1, poet_index=0, tag=None, author=None):
    result_all = []
    if poetry_file == None:
        print("Please give the poetry path!")

    if poet_index == 0:
        for index in range(0, 58000, 1000):
            shi_path = os.path.join(poetry_file, "poet.tang.%s.json" % index)
            result, _ = read_db(shi_path, tag=tag, author=author)
            result_all += result
    elif poet_index == 1:
        for index in range(0, 255000, 1000):
            shi_path = os.path.join(poetry_file, "poet.song.%s.json" % index)
            result, _ = read_db(shi_path, tag=tag, author=author)
            result_all += result
    elif poet_index == 2:
        for index in range(0, 58000, 1000):
            shi_path = os.path.join(poetry_file, "poet.tang.%s.json" % index)
            result, _ = read_db(shi_path, tag=tag, author=author)
            result_all += result
        for index in range(0, 255000, 1000):
            shi_path = os.path.join(poetry_file, "poet.song.%s.json" % index)
            result, _ = read_db(shi_path, tag=tag, author=author)
            result_all += result
    # 按诗的字数排序
    poetrys = sorted(result_all, key=lambda line: len(line))
    print(u'训练集诗总数: ', len(poetrys))

    # 统计每个字出现次数
    all_words = []
    for poetry in poetrys:
        all_words += [word for word in poetry]
    counter = collections.Counter(all_words)
    # 按照字出现的次数，从多到少对字进行排序————有利于后面对每个字的编码，从0开始，出现最多的编为0，次多的为1，依次类推
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    #  添加空白字符
    words = words + (" ", )
    # 每个字映射为一个数字ID
    word2idmap = dict(zip(words, range(len(words))))
    # 把诗转换为向量形式
    word2idfunc = lambda word: word2idmap.get(word, len(words))
    peorty_vecs = [list(map(word2idfunc, peotry)) for peotry in poetrys]

    # 准备好输入数据与标签数据，分别保存在X_data与Y_data中
    n_batch =(len(peorty_vecs)-1) // batch_size
    X_data, Y_data = [], []
    for i in range(n_batch):
        cur_vecs = peorty_vecs[i*batch_size:(i+1)*batch_size]
        current_batch_max_length = max(map(len, cur_vecs))
        batch_matrix = np.full((batch_size, current_batch_max_length), word2idfunc(" "), np.int32)
        for j in range(batch_size):
            batch_matrix[j, :len(cur_vecs[j])] = cur_vecs[j]
        x = batch_matrix
        X_data.append(x)
        y = np.copy(x)
        y[:, :-1] = x[:, 1:]
        Y_data.append(y)
    return X_data, Y_data, words, word2idfunc

if __name__=="__main__":
    # X, Y, w, w2id = get_data(os.path.join(os.getcwd(), "json"), poet_index=2)

    poetry_file = os.path.join(os.getcwd(), "json")

    result_all = []
    t = '庐山'

    for index in range(0, 58000, 1000):
        shi_path = os.path.join(poetry_file, "poet.tang.%s.json" % index)
        result, tags = read_db(shi_path, tag=t)
        result_all += result
    for index in range(0, 255000, 1000):
        shi_path = os.path.join(poetry_file, "poet.song.%s.json" % index)
        result, tags = read_db(shi_path, tag=t)
        result_all += result

    print(len(result_all))

    # with open(os.path.join(os.getcwd(),"json","tags_big.json"),"w") as tags_big_file:
    #     tags_big_file.write(json.dumps(tags_big, ensure_ascii=False))

