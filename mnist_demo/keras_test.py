#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-03-19 15:01:36
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

from keras.models import Sequential
from keras.layers.core import Dense,Activation

N_HIDDEN = 128
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape = (784,)))
model.add(Activation('relu'))
model.summary()
print(784*128)


