# -*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('UTF-8')
from collections import defaultdict


import sys
import os
import codecs

from Config import Config
from Sentence import Sentence
from Pattern import Pattern
from Seed import Seed

from gensim.matutils import cossim

import operator
from collections import defaultdict

PRINT_PATTERNS = True
'''
对实体关系进行抽取
1.定义一个Tuple类，初始的时候，对所有的sentence进行tuple化((T1,T2),(L,E1,M,E2,R))
定义类的方式参考snowball
2.初始化seed(T1,T2),根据初始的seed去匹配相应的tuple，将所有的符合的pattern找出来之后（考虑完善），
作为第一轮的pattern匹配
3 对集合里的所有Tuple进行pattern匹配，若它能与某一个pattern相匹配的话（相似度超过设定的阈值）
那么可以认为该tuple与该关系匹配，若它的关系pattern是新的话，可以纳入候选关系集合
4 对新增加的候选集合进行评估，通过确定的关系进行评估，如果达到评价标准则纳入到新的候选集中
'''

class RelationExtraction(object):


    def __init__(self,seed_file):
        self.config = Config(seed_file)
        self.patterns = list()
        self.candidate_tuples = defaultdict(list)

    def generate_tuples(self,sentences_file):
        print "\nGenerating relationship instances from sentences"
        '''
        这一步是将所有符合类型的实体对都进行了抽取（二元组关系）并存放在list中，以备后面使用
        '''
        self.sentences = Sentence(sentences_file,7,1,2,self.config.e1_type,self.config.e2_type)

    def init_bootstrapp(self):
        """
        starts a bootstrap iteration
        """
        i = 0
        # 后面全部采用参数化的方式进行迭代------>bootstrapping迭代过程
        while i <= self.config.number_iterations:
            print "\n============================================="
            print "\nStarting iteration", i
            print "\nLooking for seed matches of:"
            # 初始的实体关系种子对
            for s in self.config.seed_tuples:
                print s.e1, '\t', s.e2

            # Looks for sentences macthing the seed instances
            # 将符合该种子对的元组全部找出来
            count_matches, matched_tuples = self.match_seeds_tuples(self)
            if len(matched_tuples) == 0:
                print "\nNo seed matches found"
                sys.exit(0)
            else:
                print "\nNumber of seed matches found"
                # 将所有的种子关系对及其能匹配的元组个数打印出
                sorted_counts = sorted(count_matches.items(), key=operator.itemgetter(1), reverse=True)
                for t in sorted_counts:
                    print t[0][0], '\t', t[0][1], t[1]
                # Cluster the matched instances: generate patterns/update patterns
                # 聚类到一块的元组实例，它们称作一个簇，抽取出它们的pattern
                print "\nClustering matched instances to generate patterns"
                self.cluster_tuples(self, matched_tuples)
                # 到这一步pattern的抽取工作已经完成
                # Eliminate patterns supported by less than 'min_pattern_support' tuples
                new_patterns = [p for p in self.patterns if len(p.tuples) >= self.config.min_pattern_support]
                self.patterns = new_patterns
                for p in self.patterns:
                    print "pattern",p.__str__()
                print "\n", len(self.patterns), "patterns generated"
                if i == 0 and len(self.patterns) == 0:
                    print "No patterns generated"
                    sys.exit(0)

                # Look for sentences with occurrence of seeds semantic types (e.g., ORG - LOC)
                # This was already collect and its stored in: self.processed_tuples
                #
                # Measure the similarity of each occurrence with each extraction pattern
                # and store each pattern that has a similarity higher than a given threshold
                #
                # Each candidate tuple will then have a number of patterns that helped generate it,
                # each with an associated degree of match. Snowball uses this infer

                print "\nCollecting instances based on extraction patterns"
                count = 0
                pattern_best = None
                #对所有元组用pattern进行评估并抽取出实体对
                for t in self.sentences.relation_tuples:
                    count += 1
                    if count % 1000 == 0:
                        sys.stdout.write(".")
                        sys.stdout.flush()
                    sim_best = 0
                    #每一个元组都要用所有的pattern对它进行评估
                    for extraction_pattern in self.patterns:
                        score = self.similarity(t, extraction_pattern)
                        if score > self.config.threshold_similarity:
                            if score > self.config.extSimilar:
                                extraction_pattern.update_selectivity(t, self.config,True)
                            else:
                                extraction_pattern.update_selectivity(t, self.config,False)
                        if score > sim_best:
                            sim_best = score
                            pattern_best = extraction_pattern

                    #相似度值超过阈值的话
                    if sim_best >= self.config.threshold_similarity:
                        # if this instance was already extracted, check if it was by this extraction pattern
                        patterns = self.candidate_tuples[t]
                        #if patterns is not None:
                        if len(patterns)>0:
                            if pattern_best not in [x[0] for x in patterns]:
                                self.candidate_tuples[t].append((pattern_best, sim_best))

                        # if this instance was not extracted before, associate theisextraciton pattern with the instance
                        # and the similarity score
                        else:
                            self.candidate_tuples[t].append((pattern_best, sim_best))

                    #  update extraction pattern confidence
                    #  extraction_pattern.confidence_old = extraction_pattern.confidence
                    #  extraction_pattern.update_confidence()

                # normalize patterns confidence
                # find the maximum value of confidence and divide all by the maximum
                # 对所有的置信度归一化
                max_confidence = 0
                for p in self.patterns:
                    if p.confidence > max_confidence:
                        max_confidence = p.confidence

                if max_confidence > 0:
                    for p in self.patterns:
                        p.confidence = float(p.confidence) / float(max_confidence)

                if PRINT_PATTERNS is True:
                    print "\nPatterns:"
                    for p in self.patterns:
                        p.merge_tuple_patterns()
                        print "Patterns:", len(p.tuples)
                        print "Positive", p.positive
                        print "Negative", p.negative
                        print "Unknown", p.unknown
                        print "Tuples", len(p.tuples)
                        print "Pattern Confidence", p.confidence
                        print "\n"

                # update tuple confidence based on patterns confidence
                print "\nCalculating tuples confidence"
                for t in self.candidate_tuples.keys():
                    confidence = 1
                    t.confidence_old = t.confidence
                    for p in self.candidate_tuples.get(t):
                        confidence *= 1 - (p[0].confidence * p[1])
                    t.confidence = 1 - confidence

                    # use past confidence values to calculate new confidence
                    # if parameter Wupdt < 0.5 the system trusts new examples less on each iteration
                    # which will lead to more conservative patterns and have a damping effect.
                    if iter > 0:
                        t.confidence = t.confidence * self.config.wUpdt + t.confidence_old * (1 - self.config.wUpdt)

                # update seed set of tuples to use in next iteration
                # seeds = { T | Conf(T) > min_tuple_confidence }
                if i + 1 < self.config.number_iterations:
                    print "Adding tuples to seed with confidence =>" + str(self.config.instance_confidance)
                    for t in self.candidate_tuples.keys():
                        print "generate",t.entity1,t.entity2,t.confidence
                        if t.confidence >= self.config.instance_confidance:
                            seed = Seed(t.entity1, t.entity2)
                            print "New seed",t.entity1,t.entity2
                            self.config.seed_tuples.add(seed)

                # increment the number of iterations
                i += 1

        print "\nWriting extracted relationships to disk"
        f_output = open("relationships3.txt", "w")
        tmp = sorted(self.candidate_tuples, key=lambda tpl: tpl.confidence, reverse=True)
        for t in tmp:
            f_output.write(
                "instance: " + t.entity1.encode("utf8") + '\t' + t.entity2.encode("utf8") + '\tscore:' + str(t.confidence) + '\n')
            f_output.write("sentence: " + t.sentence.encode("utf8") + '\n')
            # writer patterns that extracted this tuple
            patterns = set()
            for pattern in self.candidate_tuples[t]:
                patterns.add(pattern[0])
            for p in patterns:
                p.merge_tuple_patterns()
                f_output.write("pattern_bet: " + ', '.join(p.tuple_patterns) + '\n')
            f_output.write("\n")
        f_output.close()

    def similarity(self, t, extraction_pattern):
        #将前中后部分分别计算相似度，并赋予相应的权重
        (bef, bet, aft) = (0, 0, 0)

        if t.before_vector is not None and extraction_pattern.centroid_bef is not None:
            bef = cossim(t.before_vector, extraction_pattern.centroid_bef)

        if t.between_vector is not None and extraction_pattern.centroid_bet is not None:
            bet = cossim(t.between_vector, extraction_pattern.centroid_bet)

        if t.after_vector is not None and extraction_pattern.centroid_aft is not None:
            aft = cossim(t.after_vector, extraction_pattern.centroid_aft)

        return self.config.alpha * bef + self.config.beta * bet + self.config.gamma * aft


    '''
    匹配所有的符合条件的Tuple
    '''
    @staticmethod
    def match_seeds_tuples(self):
        """
        checks if an extracted tuple matches seeds tuples
        """
        matched_tuples = list()
        count_matches = dict()
        for r in self.sentences.relation_tuples:
            for s in self.config.seed_tuples:
                if r.entity1 == s.e1 and r.entity2 == s.e2:
                    matched_tuples.append(r)
                    try:
                        count_matches[(r.entity1, r.entity2)] += 1
                    except KeyError:
                        count_matches[(r.entity1, r.entity2)] = 1
        return count_matches, matched_tuples

    '''
    single-pass单遍法聚类
    '''
    @staticmethod
    def cluster_tuples(self, matched_tuples):
        """
        single-pass clustering
        """
        start = 0
        # Initialize: if no patterns exist, first tuple goes to first cluster
        # 初始化时第一个元祖的向量自成一个pattern
        if len(self.patterns) == 0:
            c1 = Pattern(matched_tuples[0])
            self.patterns.append(c1)
            start = 1

        # Compute the similarity between an instance with each pattern go through all tuples
        # 继续对接下来的tuples进行聚类，如果跟前面的某一个相似度最高，将其归入那一个簇中，并更新pattern
        for i in range(start, len(matched_tuples), 1):
            t = matched_tuples[i]
            max_similarity = 0
            max_similarity_cluster_index = 0

            # go through all patterns(clusters of tuples) and find the one with the
            # highest similarity score
            for w in range(0, len(self.patterns), 1):
                extraction_pattern = self.patterns[w]
                score = self.similarity(t, extraction_pattern)
                if score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = w

            # if max_similarity < min_degree_match create a new cluster having this tuple as the centroid
            if max_similarity < self.config.threshold_similarity:
                c = Pattern(t)
                self.patterns.append(c)

            # if max_similarity >= min_degree_match add to the cluster with the highest similarity
            else:
                self.patterns[max_similarity_cluster_index].add_tuple(t)

def main():
    re = RelationExtraction(seed_file='seedfile3.txt')
    re.generate_tuples('text.txt')
    re.init_bootstrapp()

if __name__ == "__main__":
    main()