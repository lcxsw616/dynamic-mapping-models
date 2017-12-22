#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import loader
import network

print ("开始训练")

training_data, validation_data, test_data = loader.load_data_decode()
# 784 个输入神经元，一层隐藏层，包含 30 个神经元，输出层包含 10 个神经元

net = network.Network([784, 20, 20, 10])
net.SGD(training_data, 20, 20, 2.0, test_data = test_data)
# Epoch 0: 9038 / 10000
# Epoch 1: 9178 / 10000
# Epoch 2: 9231 / 10000
# ...
# Epoch 27: 9483 / 10000
# Epoch 28: 9485 / 10000
# Epoch 29: 9477 / 10000