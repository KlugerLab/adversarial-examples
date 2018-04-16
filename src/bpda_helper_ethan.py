import matlab.engine
import tensorflow as tf
import numpy as np
import os
from scipy.misc import imsave, imread
import inceptionv3
from utils import load_image, one_hot
import subprocess

slim = tf.contrib.slim
model_path = ""
data_path = ""

def load_images(image_list, imgdir_path, imgdir_denoised, batch_shape, img_fmt):
    images = np.zeros(batch_shape)
    images_denoised = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in image_list:
        with tf.gfile.Open(os.path.join(imgdir_path, filepath), "rb") as f:
            images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float)*2.0/255.0 - 1.0
        if img_fmt == 'jpg':
            filepath = filepath.replace('png', img_fmt)
        with tf.gfile.Open(os.path.join(imgdir_denoised, filepath), "rb") as f:
            images_denoised[idx, :, :, :] = imread(f, mode='RGB').astype(np.float)*2.0/255.0 - 1.0
        if img_fmt == 'jpg':
            filepath = filepath.replace(img_fmt, 'png')
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images_denoised, images
            filenames = []
            images = np.zeros(batch_shape)
            images_denoised = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images_denoised, images

def bpda_attack(defense, defense_name, attack_save_dir, image_list, config, **kwargs):
    os.environ['CUDA_VISIBLE_DEVICES'] =  config.gpu_num
    print('kwargs')
    print(kwargs)
    base_net = config.base_net
    if base_net == 'inception_v3':
        from tensorflow.contrib.slim.nets import inception
        from model import InceptionModel as Model
        image_width       = 299
        image_height      = 299
        checkpoint_path   = model_path + "/inception_v3.ckpt"
    elif base_net == 'renet':
        from tensorflow.contrib.slim.nets import resnet
        from model import ResnetModel
    else:
        print("Specify model") 

    if defense == 'jpeg':
        img_fmt = 'jpg'
    else:
        img_fmt = 'png'

    eps_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    for eps in eps_list:
        from cleverhans.attacks import BasicBPDAMethod as AdvMethod
        adv_args = {'eps': eps, 'clip_min':-1., 'clip_max':1., 'nb_iter':1}
        benign_dir        = data_path + "/attack/benign"
        denoised_benign_dir = data_path + "/DenoisedSamples/"+defense_name+"/benign"
        save_dir_path = attack_save_dir+'/'+defense_name+'/eps'+str(eps) + '/'
        if not os.path.exists(save_dir_path):
            subprocess.check_call(['mkdir', '-p', save_dir_path]) 
	
        print('running BPDA attack...')
        batch_shape = config.batch_shape
        num_classes = 1001
        with tf.Graph().as_default():
            with tf.Session() as sess:
                x_benign = tf.placeholder(tf.float32, shape=batch_shape)
                x_denoised = tf.placeholder(tf.float32, shape=batch_shape)
                model = Model(num_classes)
                adv_method  = AdvMethod(model, sess=sess)
                x_adv = adv_method.generate(x_denoised, x_benign, **adv_args)

                predicted_labels, nb_classes = adv_method.get_or_guess_labels(x_benign, {})
                predicted_labels = tf.argmax(predicted_labels, 1)
                print(nb_classes)

                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(slim.get_model_variables())
                saver.restore(sess, checkpoint_path)

                for filenames, denoised_images, images in load_images(image_list, benign_dir, denoised_benign_dir, batch_shape, img_fmt):
                    print(len(filenames))
                    print(np.linalg.norm(denoised_images[0] - images[0]))
                    nontargeted_images = sess.run(x_adv, feed_dict={x_denoised: denoised_images, x_benign : images})
                    save_images(nontargeted_images, filenames, save_dir_path)

def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        img = np.uint8(((images[i, :, :, :] + 1.0) * 0.5) * 255.0)
        imsave(output_dir+filename, img)

