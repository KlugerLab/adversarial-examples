from jpeg_denoise import jpeg
from pca_whole_denoise import pca_whole
from pca_blockwise_denoise import pca_blockwise
from softthresh_denoise import softthresh
from lv1approx_denoise import lv1approx
from fourier_blockwise_denoise import fourier_blockwise
from fourier_whole_denoise import fourier_whole
from denoise_wrapper import denoise_wrapper
from bpda_helper_ethan import bpda_attack

data_path = ""

def run_bpda_attack(image_list,config):
    print('running defense...')
    tuples           = config.create_tuples()
    subset_num       = config.subset_num
    random_seed      = config.random_seed
    attack_base_dir  = data_path + "attack/"
    for (attack_name, defense_args) in tuples:
        defense_args_split  = defense_args.split('_')
        defense_name  = defense_args_split[0]

        if defense_name == 'jpeg':
            if len(defense_args_split) < 2:
                print("No approximation level provided for jpeg.  Skipping...")
            else:    
                approx_level = int(defense_args_split[1])
                attack_dir = attack_base_dir + attack_name
                bpda_attack(defense_name, defense_args, attack_dir, image_list, config, approx_level=approx_level)
        elif defense_name == 'pca-whole':
            if len(defense_args_split) < 2:
                print("Number of components missing for PCA.  Skipping...")
            else:
                num_components = int(defense_args_split[1])
                attack_dir = attack_base_dir + attack_name
                bpda_attack(defense_name, defense_args, attack_dir, image_list, config, num_components=num_components)
        elif defense_name == 'pca-patch':
            if len(defense_args_split) < 2:
                print("Number of components missing for PCA.  Skipping...")
            else:
                num_components = int(defense_args_split[1])
                attack_dir = attack_base_dir + attack_name
                bpda_attack(defense_name, defense_args, attack_dir, image_list, config, num_components=num_components)
        elif defense_name == 'softthresh':
            if len(defense_args_split) < 3:
                print("Multiplicative constant missing or/and wavelet function for softthresh.  Skipping...")
            else:    
                mul_thresh = defense_args_split[1]
                wavelet_func = defense_args_split[2]
                attack_dir = attack_base_dir + attack_name
                bpda_attack(defense_name, defense_args, attack_dir, image_list, config, mul_thresh=mul_thresh, wavelet_func=wavelet_func)
        elif defense_name == 'lv1approx':
            if len(defense_args_split) < 2:
                print("Missing wavelet function for lv1approx.  Skipping...")
            else:    
                wavelet_func = defense_args_split[1]
                attack_dir = attack_base_dir + attack_name
                bpda_attack(defense_name, defense_args, attack_dir, image_list, config, wavelet_func=wavelet_func)
        elif defense_name == 'fourier-whole':
            if len(defense_args_split) < 2:
                print("Missing radius for Fourier filtering. Skipping...")
            else:
                lowPassRadius = defense_args_split[1]
                highPassRadius = defense_args_split[2]  
                attack_dir = attack_base_dir + attack_name
                bpda_attack(defense_name, defense_args, attack_dir, image_list, config, lowPassRadius=lowPassRadius, highPassRadius=highPassRadius)
        else:
            print("Defense not recognized.  Skipping...")

if __name__=='__main__':
    run_defense()

