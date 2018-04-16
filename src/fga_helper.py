import matlab.engine
import tensorflow as tf
import numpy as np
import os
from scipy.misc import imsave, imread
import inceptionv3
from utils import load_image, one_hot
from fga_wrapper import fga_wrapper 
import scipy.io as sio

data_path = ""

def get_benign_images():
    input_dir = data_path + "/attack/benign/"
    image_height=299
    image_width=299
    num_images = 1000
    batch_shape = [num_images, image_height, image_width, 3]
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    for filepath in sorted(tf.gfile.Glob(os.path.join(input_dir, '*.png'))):
        with tf.gfile.Open(filepath, "rb") as f:
            images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float)*2.0/255.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_shape[0]:
            break
    return filenames, images

def fga_attack(defense, defense_name, attack_save_dir, config, **kwargs):
    os.environ['CUDA_VISIBLE_DEVICES'] =  config.gpu_num
    print(kwargs)
    filenames, benign_images = get_benign_images()
    
    print('running filtered gradient attack...')

            
    if defense in {'softthresh','lv1approx', 'fourier-whole','pca-patch'}:
        eng = matlab.engine.start_matlab()
                
    if defense == 'softthresh':
        thresh_mul = kwargs['mul_thresh']
        wavelet_func = kwargs['wavelet_func']
        _ = eng.fga_wrapper(defense, thresh_mul, wavelet_func)
    elif defense == 'lv1approx':
        wavelet_func = kwargs['wavelet_func']
        _ = eng.fga_wrapper(defense, wavelet_func)
    elif defense == 'fourier-whole':
        lowPassRadius = kwargs['lowPassRadius']
        highPassRadius = kwargs['highPassRadius']
        _ = eng.fga_wrapper(defense, lowPassRadius, highPassRadius)
    elif defense == 'pca-patch':
        num_components = kwargs['num_components']
        _ = eng.fga_wrapper(defense, num_components)
    elif defense == 'pca-whole':
        num_components = kwargs['num_components']
        denoised_grad = fga_wrapper(defense, num_components=num_components)    
    elif defense == 'jpeg':
        approx_level = kwargs['approx_level']
        denoised_grad = fga_wrapper(defense, approx_level=approx_level) 
    else:
        pass
    if defense in {'softthresh','lv1approx', 'fourier-whole','pca-patch'}:
        print('loading denoised_grad.mat...')
        mat_contents = sio.loadmat('denoised_grad.mat')
        denoised_grad = mat_contents['X_denoised']
        print('loading denoised_grad.mat done')


    l2_desired_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    lv1_desired_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    lv1_desired_list = [0.005, 0.0075, 0.01, 0.0125, 0.015] 

    for desired_l2, lv1_desired in zip(l2_desired_list, lv1_desired_list): 
        save_dir_path = attack_save_dir+'/'+defense_name+'/eps'+str(desired_l2)
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        adv = benign_images + lv1_desired * denoised_grad 
        adv = np.clip(adv, -1, 1)
        save_images(adv, filenames, save_dir_path) 

def save_images(images, filenames, save_dir_path): 
    for filename, image in zip(filenames, images):
        img = np.uint8(((image + 1.0) * 0.5) * 255.0)
        imsave(save_dir_path+'/'+filename, img)



