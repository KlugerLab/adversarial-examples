import matlab.engine
from PIL import Image
import os
import numpy as np
import subprocess 
from scipy.misc import imread, imsave
from sklearn.decomposition import PCA
from skimage.util import view_as_blocks

def get_subdirs(base_dir):
        return list(filter(os.path.isdir, [os.path.join(base_dir,sub_dir) for sub_dir in os.listdir(base_dir)]))

def denoise_img_whole_pca(image_path, n_comp):
    with open(image_path, "rb") as f:
        img = imread(f, mode='RGB').astype(np.float)*2.0/255.0 - 1.0
    img_restored = np.empty_like(img)
    for colorNum in range(0, 3): #Color channels
        img_onecolor = img[:,:,colorNum]

        pca = PCA(n_components = n_comp)
        pca.fit(img_onecolor)
        onecolor_pca = pca.fit_transform(img_onecolor)

        img_onecolor_restored = pca.inverse_transform(onecolor_pca)
        img_restored[:,:,colorNum] = img_onecolor_restored
    return img_restored

def denoise_img_block_pca(image_path, n_comp, block_shape):
    with open(image_path, "rb") as f:
        img = imread(f, mode='RGB').astype(np.float)*2.0/255.0 - 1.0
    img_restored = np.empty_like(img)
    for colorNum in range(0, 3): #Color channels
        img_onecolor = img[:,:,colorNum]
        img_onecolor_blocks = view_as_blocks(img_onecolor, block_shape)
        img_onecolor_restored_blocks = np.empty_like(img_onecolor_blocks)

        shape = img_onecolor_blocks.shape
        for (k, l) in np.ndindex((shape[2], shape[3])):
            block = img_onecolor_blocks[:,:,k,l]
            pca = PCA(n_components = n_comp)


            pca.fit(block)
            block_pca = pca.fit_transform(block)
            img_onecolor_restored_blocks[:,:,k,l] = pca.inverse_transform(block_pca)
        x = img_onecolor_restored_blocks.transpose(0,2,1,3).reshape(299,299)
        img_restored[:,:,colorNum] = x
    return img_restored

def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
            .swapaxes(1,2)
            .reshape(-1, nrows, ncols))

def denoise_wrapper(method, defense_args, attack_dir, denoise_base_dir, image_list, **kwargs):
    #Get name of attack (e.g. FGM)..
    attack_name = os.path.basename(os.path.normpath(attack_dir))

    #Directory where adv images live
    if attack_name in {'FGM', 'I-FGM', 'CW'}:
        l2_dir_list = sorted(get_subdirs(attack_dir))
    elif attack_name in {'BPDA', 'BPDA_new', 'FGA'}:
        print(attack_dir)
        l2_dir_list = sorted(get_subdirs(attack_dir+'/'+defense_args))
    else:
        print('Error: attack_name is invalid.')
        import sys;sys.exit()

    #Loop through every folder of the form l2dis_0.0x
    for l2_dir in l2_dir_list:
        images      = image_list
        l2_distance = os.path.basename(os.path.normpath(l2_dir))

        # Denoise by each method
        if method == 'jpeg':
            approx_level = kwargs['approx_level']
            output_dir = denoise_base_dir + method + '_' + str(approx_level) + '/' + attack_name + '/' + l2_distance + '/'

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for image in images:
                im1 = Image.open(os.path.join(l2_dir, image))
                filename, file_extension = os.path.splitext(image)
                jpeg_filename = filename + ".jpg"
                IMAGE_10 = os.path.join(output_dir, jpeg_filename)
                im1.save(IMAGE_10,"JPEG", quality=approx_level)
                im10 = Image.open(IMAGE_10)
        elif method == 'pca-whole':
            num_components = kwargs['num_components']
            output_dir = denoise_base_dir + method + '_' + str(num_components) + '/' + attack_name + '/' + l2_distance + '/'             
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            #for (image, path) in zip(images, paths):
            for image in images:
                denoised_img = denoise_img_whole_pca(os.path.join(l2_dir,image), num_components)
                imsave(output_dir + image, denoised_img)
        elif method == 'pca-patch':
            num_components = kwargs['num_components']
            output_dir = denoise_base_dir + method + '_' + str(num_components) + '/' + attack_name + '/' + l2_distance + '/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            eng = matlab.engine.start_matlab()
            _ = eng.denoise_wrapper(l2_dir, images, output_dir, method, num_components)
        elif method == 'pca-blockwise': 
            num_components = kwargs['num_components']
            block_shape    = (23, 23) # There are only two options: (13, 13) or (23,23) so hard-coded
            output_dir = denoise_base_dir + method + '_' + str(num_components) + '/' + attack_name + '/' + l2_distance + '/'             
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for image in images:
                denoised_img = denoise_img_block_pca(os.path.join(l2_dir,image), num_components, block_shape)
                imsave(output_dir + image, denoised_img)
        elif method == 'softthresh':
            # get parameters
            mul_thresh = kwargs['mul_thresh']
            wavelet_func = kwargs['wavelet_func']
            output_dir = denoise_base_dir + method + '_' + str(mul_thresh) + '_' + wavelet_func + '/' + attack_name + '/' + l2_distance + '/'             
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Run the matlab version of denoise_wrapper.m and pass image_list there 
            # Use matlab.engine to run denoise_wrapper.m
            # Inside the denoise_wrapper.m, loop through images and call {method}.m
            # for each image. 
            eng = matlab.engine.start_matlab()
            _ = eng.denoise_wrapper(l2_dir, images, output_dir, method, mul_thresh, wavelet_func)
    
        elif method == 'lv1approx':
            wavelet_func = kwargs['wavelet_func']
            output_dir = denoise_base_dir + method + '_'  + wavelet_func + '/' + attack_name + '/' + l2_distance + '/'             
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            eng = matlab.engine.start_matlab()
            _ = eng.denoise_wrapper(l2_dir, images, output_dir, method, wavelet_func)
        elif method == 'fourier-whole':    
            lowPassRadius = kwargs['lowPassRadius']
            highPassRadius = kwargs['highPassRadius']
            output_dir = denoise_base_dir + method + '_'  + str(lowPassRadius) +'_' +str(highPassRadius)+ '/' + attack_name + '/' + l2_distance + '/'             
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            eng = matlab.engine.start_matlab()
            _ = eng.denoise_wrapper(l2_dir, images, output_dir, method, lowPassRadius, highPassRadius)
        elif method == 'fourier-blockwise':    
            lowPassRadius = kwargs['lowPassRadius']
            highPassRadius = kwargs['highPassRadius']
            output_dir = denoise_base_dir + method + '_'  + str(lowPassRadius) +'_'+ str(highPassRadius) + '/' + attack_name + '/' + l2_distance + '/'             
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            eng = matlab.engine.start_matlab()
            _ = eng.denoise_wrapper(l2_dir, images, output_dir, method, lowPassRadius, highPassRadius)
        else:
            print('Error: denoise method is not defined. Skipping...')
            
