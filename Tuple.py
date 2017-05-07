#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.setdefaultencoding('UTF-8')

import sys
import os
import codecs

class Tuple(object):
    def __init__(self, _e1, _e2, _type1,_type2,_before,_between,_after,_sentence):
        self.entity1 = _e1
        self.entity2 = _e2
        self.ent_type1 = _type1
        self.ent_type2 = _type2
        self.sentence = _sentence

        self.confidence = 0
        self.confidence_old = 0

        self.before_semantic = [e[0] for e in _before]
        self.bef_words = " ".join(self.before_semantic)
        self.before_vector = [(e[1],e[2]) for e in _before]
        self.between_semantic = [e[0] for e in _between]
        self.between_vector = [(e[1],e[2]) for e in _between]
        self.bet_words = " ".join(self.between_semantic)
        self.after_semantic = [e[0] for e in _after]
        self.after_vector = [(e[1],e[2]) for e in _after]
        self.aft_words = " ".join(self.after_semantic)

    def __str__(self):
        return str(self.bef_words.encode("utf8") + ' ' + self.bet_words.encode("utf8") + ' ' +
                   self.aft_words.encode("utf8"))

    def __eq__(self, other):
        return (self.entity1 == other.entity1 and self.entity2 == other.entity2 and self.bef_words == other.bef_words and
                self.bet_words == other.bet_words and self.aft_words == other.aft_words)

    def get_vector(self, context):
        if context == "bef":
            return self.before_vector
        elif context == "bet":
            return self.between_vector
        elif context == "aft":
            return self.after_vector
        else:
            print "Error, vector must be 'bef', 'bet' or 'aft'"
            sys.exit(0)
