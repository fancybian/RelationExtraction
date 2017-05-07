# -*- coding: UTF-8 -*-
import sys
from Tuple import Tuple
reload(sys)
sys.setdefaultencoding('UTF-8')

import sys
import os
import codecs

from gensim import corpora, similarities, models
'''
1.初始时需要对停用词进行剔除（待完成）
2.纳入到tfidf模型中，对每一个词进行tfidf计算
3.补充一个try catch
4.将每一个词对应出来

'''


class Sentence:
    '''
    初始化的开始就应该率先将tf-idf权重的值算出来，然后对应到每一个词上
    '''
    def __init__(self, _sentences, max_tokens, min_tokens, window_size,ent1type,ent2type):
        self.relation_tuples = []
        #read all raw sentences
        f_sentences = codecs.open(_sentences, encoding='utf-8')
        #put all entities in seg list by line
        corpora_documents = []
        #read sentence by line
        for line in f_sentences:
            st_entities = line.split()
            item_seg = [word.split('/')[0] for word in st_entities]
            corpora_documents.append(item_seg)

        # 生成字典和向量语料
        dictionary = corpora.Dictionary(corpora_documents)
        # 通过下面一句得到语料中每一篇文档对应的稀疏向量（这里是bow向量）
        corpus = [dictionary.doc2bow(text) for text in corpora_documents]
        # 向量的每一个元素代表了一个word在这篇文档中出现的次数
        #print(corpus)
        # corpus是一个返回bow向量的迭代器。下面代码将完成对corpus中出现的每一个特征的IDF值的统计工作
        tfidf_model = models.TfidfModel(corpus)
        corpus_tfidf = tfidf_model[corpus]
        '''
        由于上面的统计中已经把tfidf统计出来了，所以下面就直接用它进行关系抽取
        '''
        #获取每一个word的id编号
        token2id = dictionary.token2id
        #读取每一句话
        f_sentences = codecs.open(_sentences, encoding='utf-8')
        r = 0
        cnt = 0
        for line in f_sentences:
            k = 0
            matches = []
            st_entities = line.split()
            for entity in st_entities:
                if entity.split('/')[1] != 'o':
                    matches.append((entity.split('/')[0], entity.split('/')[1], k))
                k += 1
            #存放的是key:id和value:weights(tf-idf计算出来的)
            id2weights = {}
            #print corpus_tfidf[r]
            for x in corpus_tfidf[r]:
                id2weights[x[0]] = x[1]
            '''
            将每一个词及其权重对应起来
            '''
            for x in range(len(st_entities)):
                token_name = st_entities[x].split('/')[0]
                #print token_name
                #print token2id[token_name]
                try:
                    tfidf = id2weights[token2id[token_name]]
                except KeyError:
                    tfidf = 0
                st_entities[x] = (st_entities[x].split('/')[0],token2id[token_name], tfidf)
            '''
            将所有的实体关系对找到，并且生成一个个Tuple
            '''
            if len(matches) >= 2:
                for x in range(0, len(matches) - 1):
                    idx1 = matches[x][2]
                    idx2 = matches[x + 1][2]
                    argtype1 = matches[x][1]
                    argtype2 = matches[x + 1][1]
                    if idx2 - idx1 > min_tokens and idx2 - idx1 < max_tokens and argtype1 == ent1type and argtype2 == ent2type:
                        ent1 = matches[x][0]
                        ent2 = matches[x + 1][0]
                        between = st_entities[idx1 + 1:idx2]
                        before = st_entities[:idx1][-window_size:]
                        after = st_entities[idx2 + 1:][:window_size]
                        #print ent1, argtype1, ent2, argtype2, before, between, after
                        cnt += 1
                        self.relation_tuples.append(Tuple(ent1,ent2,argtype1,argtype2,before,between,after,line))
            r += 1
        print "total;"+str(cnt)


