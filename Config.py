#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fileinput
import os
import sys
from Seed import Seed
class Config(object):

    def __init__(self, seeds_file):

        self.seed_tuples = set()
        self.read_seeds(seeds_file)
        self.alpha = 0.1
        self.beta = 0.8
        self.gamma = 0.1
        self.min_pattern_support = 2
        self.threshold_similarity = 0.7
        self.wUnk = 0.1
        self.wNeg = 0.9
        self.number_iterations = 5
        self.wUpdt = 0.5
        self.extSimilar = 0.8

        self.instance_confidance = 0.6

    def read_seeds(self, seeds_file):
        for line in fileinput.input(seeds_file):
            if line.startswith("#") or len(line) == 1:
                continue
            if line.startswith("e1"):
                self.e1_type = line.split(":")[1].strip()
            elif line.startswith("e2"):
                self.e2_type = line.split(":")[1].strip()
            else:
                e1 = line.split(";")[0].strip()
                e2 = line.split(";")[1].strip()
                seed = Seed(e1, e2)
                self.seed_tuples.add(seed)