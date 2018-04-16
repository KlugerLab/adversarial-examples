import numpy as np
import itertools


class Config(object):
    def __init__(self):
        self.subset_num = 1000 #16 #1000 #32
        self.random_seed = None 
        self.gpu_num    = '0'
        self.batch_shape = [16, 299, 299, 3] # attack.py and evaluate.py depends on this
        self.base_net   = 'inception_v3'
        self.attack_list = ['FGM', 'I-FGM', 'CW']
        self.attack_list = ['FGA']
        self.defense_list = ['fourier-whole_65_0','pca-whole_36','softthresh_1.0_bior3.1','lv1approx_bior3.1', 'jpeg_23', 'pca-patch_13']
        self.setting = 'graybox' #'blackbox'
        # self.setting = 'whitebox' #'blackbox'

    def create_tuples(self):
        tuple_list = []
        if self.setting == 'blackbox' or self.setting == 'graybox':
            for i in itertools.product(self.attack_list, self.defense_list):
                tuple_list.append(i)
        elif self.setting == 'whitebox':
            for i in range(len(self.defense_list)):
                tuple_list.append((self.attack_list[0], self.defense_list[i]))

        return tuple_list
        
