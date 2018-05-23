# coding = utf-8
import collections
import numpy as np
import codecs

# 数据读取与预处理,先用一个比较简单的数据集
def get_data(poetry_file = 'data/poetry.txt',batch_size = 1):
    # 诗集
    poetrys = []
    with codecs.open(poetry_file, "r", 'utf-8') as f:
        for line in f:
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = '[' + content + ']'
                poetrys.append(content)
            except Exception as e:
                print(e)

    # 按诗的字数排序
    poetrys = sorted(poetrys, key=lambda line: len(line))
    print(u'唐诗总数: ', len(poetrys))  # 3w多首诗

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
    X,Y,w,w2id = get_data()