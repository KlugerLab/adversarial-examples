from fga_helper import fga_attack 

def run_fga_attack(config):
    print('running fga attack...')
    tuples           = config.create_tuples()
    subset_num       = config.subset_num
    random_seed      = config.random_seed
    attack_base_dir  = "/data/yutaro/dataset/attack/"
    for (attack_name, defense_args) in tuples:
        defense_args_split  = defense_args.split('_')
        defense_name  = defense_args_split[0]

        if defense_name == 'jpeg':
            if len(defense_args_split) < 2:
                print("No approximation level provided for jpeg.  Skipping...")
            else:    
                approx_level = int(defense_args_split[1])
                attack_dir = attack_base_dir + attack_name
                fga_attack(defense_name, defense_args, attack_dir, config, approx_level=approx_level)
        elif defense_name == 'pca-whole':
            if len(defense_args_split) < 2:
                print("Number of components missing for PCA.  Skipping...")
            else:
                num_components = int(defense_args_split[1])
                attack_dir = attack_base_dir + attack_name
                fga_attack(defense_name, defense_args, attack_dir, config, num_components=num_components)
        elif defense_name == 'pca-patch':
            if len(defense_args_split) < 2:
                print("Number of components missing for PCA.  Skipping...")
            else:
                num_components = int(defense_args_split[1])
                attack_dir = attack_base_dir + attack_name
                fga_attack(defense_name, defense_args, attack_dir, config, num_components=num_components)
        elif defense_name == 'softthresh':
            if len(defense_args_split) < 3:
                print("Multiplicative constant missing or/and wavelet function for softthresh.  Skipping...")
            else:    
                mul_thresh = defense_args_split[1]
                wavelet_func = defense_args_split[2]
                attack_dir = attack_base_dir + attack_name
                fga_attack(defense_name, defense_args, attack_dir, config, mul_thresh=mul_thresh, wavelet_func=wavelet_func)
        elif defense_name == 'lv1approx':
            if len(defense_args_split) < 2:
                print("Missing wavelet function for lv1approx.  Skipping...")
            else:    
                wavelet_func = defense_args_split[1]
                attack_dir = attack_base_dir + attack_name
                fga_attack(defense_name, defense_args, attack_dir, config, wavelet_func=wavelet_func)
        elif defense_name == 'fourier-whole':
            if len(defense_args_split) < 2:
                print("Missing radius for Fourier filtering. Skipping...")
            else:
                lowPassRadius = defense_args_split[1]
                highPassRadius = defense_args_split[2]  
                attack_dir = attack_base_dir + attack_name
                fga_attack(defense_name, defense_args, attack_dir, config, lowPassRadius=lowPassRadius, highPassRadius=highPassRadius)
        else:
            print("Defense not recognized.  Skipping...")

if __name__=='__main__':
    run_fga_attack()

