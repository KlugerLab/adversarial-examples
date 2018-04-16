import glob
import os
import numpy as np

from scipy.misc import imread, imsave
from sklearn.decomposition import PCA
from skimage.util import view_as_blocks

def get_subdirs(base_dir):
    return list(filter(os.path.isdir, [os.path.join(base_dir,sub_dir) for sub_dir in os.listdir(base_dir)]))

def load_images(input_dir):
    image_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    
    num_images = len(image_paths)
    image_height = 299
    image_width  = 299
    num_channels = 3
    
    images = np.zeros([num_images, image_height, image_width, num_channels])
    filenames = []
    idx = 0

    for image_path in image_paths:
        with open(image_path, "rb") as f:
            images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float)*2.0/255.0 - 1.0
        filenames.append(os.path.basename(image_path))
        idx += 1
    return images, filenames

def denoise_img(img, n_comp):
    img_restored = np.empty_like(img)
    for colorNum in range(0, 3): #Color channels
        img_onecolor = img[:,:,colorNum]
        
        pca = PCA(n_components = n_comp)
        pca.fit(img_onecolor)
        onecolor_pca = pca.fit_transform(img_onecolor)

        img_onecolor_restored = pca.inverse_transform(onecolor_pca)
        img_restored[:,:,colorNum] = img_onecolor_restored
    return img_restored

def pca_whole(attack_dir, denoise_base_dir, num_components):
    attack_name = os.path.basename(os.path.normpath(attack_dir))
    l2_dir_list = sorted(get_subdirs(attack_dir))

    for l2_dir in l2_dir_list:
        images, paths = load_images(l2_dir)
        l2_distance   = os.path.basename(os.path.normpath(l2_dir))
        output_dir    =  denoise_base_dir + pca_whole.__name__ + '_' + str(num_components) + '/' + attack_name + '/' + l2_distance + '/'
        print(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for (image, path) in zip(images, paths):
            denoised_img = denoise_img(image, num_components) 
            imsave(output_dir + path, denoised_img)
