import subprocess 
import os 
import datetime

def get_subdirs(base_dir):
    return list(filter(os.path.isdir, [os.path.join(base_dir,sub_dir) for sub_dir in os.listdir(base_dir)])) 


def fourier_whole(attack_dir, denoise_base_dir, lowPassRadius, highPassRadius):
    #Get name of attack (e.g. FGM)
    attack_name = os.path.basename(os.path.normpath(attack_dir))

    #Directory where adv images live
    l2_dir_list = sorted(get_subdirs(attack_dir))

    #Loop through every folder of the form l2dis_0.0x
    for l2_dir in l2_dir_list:
        l2_distance = os.path.basename(os.path.normpath(l2_dir))
        output_dir  =  denoise_base_dir + fourier_whole.__name__ + '_' + str(lowPassRadius) + '/' + attack_name + '/' + l2_distance + '/'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # run the matlab thing here
        input_dir_adv = l2_dir
        script_name = 'denoise_fourier_whole' # if args exits: 'test Arg1 Arg2'
        cmd_name = script_name + ' ' + input_dir_adv + '/ ' + output_dir + ' ' + str(lowPassRadius) + ' ' + str(highPassRadius)
        cmd = "matlab -nodisplay -r '" + cmd_name + "; exit;'"
        subprocess.call(cmd, shell=True)
