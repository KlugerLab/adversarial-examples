import matlab.engine 
import numpy as np
from config import Config
import subprocess
import random
import os
from defense import run_defense
from evaluate import run_eval
#from bpda import bpda_attack
from run_bpda_attack import run_bpda_attack
from run_fga_attack import run_fga_attack

config           = Config()
dataset_path     = "/data/yutaro/dataset/"
attack_base_dir  = dataset_path + "attack/"
denoise_base_dir = dataset_path + "DenoisedSamples/"
benign_dir        = dataset_path + "attack/benign"

def get_image_list(benign_dir, random_seed, subset_num):
    images_whole      = [f for f in os.listdir(benign_dir) if os.path.isfile(os.path.join(benign_dir, f))]
    images_whole      = sorted(images_whole)

    if random_seed != None:
        random.seed(random_seed)
        subset_ids  = random.sample(range(len(images_whole)), subset_num)  # generate random indices to subset
    else:
        subset_ids  = range(subset_num)

    image_list = [images_whole[i] for i in subset_ids]
    return image_list


def main(): 
    random_seed = config.random_seed
    subset_num  = config.subset_num
    image_list = get_image_list(benign_dir, random_seed, subset_num)
    #run_attack(image_list) # TODO: image_list needs to be passed into attack.py
    #run_defense(image_list,config) 
    #run_eval(image_list,config)
    #run_bpda_attack(image_list, config)
    run_fga_attack(config)
    run_defense(image_list,config) 
    run_eval(image_list,config)
    

main()
