#!/usr/bin/env python
# encoding: utf8
#
#       @author       : tcund@126.com
#       @file         : dict_deque.py
#       @date         : 2018/05/03 18:50
from collections import deque

class DictDeque(object):
    def __init__(self, maxlen):
        self.__que = deque()
        self.__dict = dict()
        self.__maxlen = maxlen

    def get(self, key, default = None):
        if key.__hash__ is None:
            key = str(key)
        return self.__dict.get(key, default)

    def has(self, key):
        if key.__hash__ is None:
            key = str(key)
        return key in self.__dict

    def set(self, key, value):
        if key.__hash__ is None:
            key = str(key)
        if not self.has(key):
            if len(self.__que) >= self.__maxlen:
                del_key = self.__que.popleft()
                del self.__dict[del_key]
            self.__que.append(key)
        self.__dict[key] = value

    def clear(self):
        self.__que.clear()
        self.__dict.clear()

if __name__ == '__main__':
    pass
