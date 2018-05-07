#coding=utf-8

import collections
import numpy as np
import tensorflow as tf
import codecs
import os

#数据读取与预处理
poetry_file ='data/draft.txt'
# 诗集
poetrys = []
with codecs.open(poetry_file, "r",'utf-8') as f:
	for line in f:
		try:
			title, content = line.strip().split(':')
			content = content.replace(' ','')
			if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
				continue
			if len(content) < 5 or len(content) > 79:
				continue
			content = '[' + content + ']'
			poetrys.append(content)
		except Exception as e: 
			print (e)
# 按诗的字数排序,从少到多
poetrys = sorted(poetrys,key=lambda line: len(line))
print("peotrys:",poetrys)
print(u'唐诗总数: ', len(poetrys))#3w多首诗

# 统计每个字出现次数
all_words = []
for poetry in poetrys:
	all_words += [word for word in poetry]
print("all_words:",all_words)
counter = collections.Counter(all_words)
print ("counter:",counter)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
print("counter_pairs:",count_pairs)
words, _ = zip(*count_pairs)
print("words:",words)
#add empty char
words = words + (" ",) 
# map word to id
# 每个字映射为一个数字ID
word2idmap = dict(zip(words,range(len(words))))
print("word2idmap:",word2idmap)
# 把诗转换为向量形式
word2idfunc = lambda word:  word2idmap.get(word,len(words))
peorty_vecs  = [list(map(word2idfunc,peotry)) for peotry in poetrys]
print(peorty_vecs)