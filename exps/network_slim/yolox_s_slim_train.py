#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.max_epoch = 300
        self.network_slim_sparsity_train_enable = True
        self.network_slim_sparsity_train_s = 0.0001
        self.network_slim_sparsity_train_warmup_epoch = 120

        self.data_dir = "/local_datasets/VisDrone"
        self.train_ann = "train2017.json"
        self.val_ann = "val2017.json"
        self.test_ann = "test2017.json"
        self.input_size = (832, 832)
        self.test_size = (832, 832)

        self.num_classes = 10
