#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import codecs

sys.setdefaultencoding('UTF-8')


class Seed(object):
    def __init__(self, _e1, _e2):
        self.e1 = _e1
        self.e2 = _e2

    def __hash__(self):
        return hash(self.e1) ^ hash(self.e2)

    def __eq__(self, other):
        return self.e1 == other.e1 and self.e2 == other.e2