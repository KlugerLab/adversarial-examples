from PIL import Image
import os


def get_subdirs(base_dir):
    return list(filter(os.path.isdir, [os.path.join(base_dir,sub_dir) for sub_dir in os.listdir(base_dir)]))

def jpeg(attack_dir, denoise_base_dir, approx_level):
    #Get name of attack (e.g. FGM)
    attack_name = os.path.basename(os.path.normpath(attack_dir))

    #Directory where adv images live
    l2_dir_list = sorted(get_subdirs(attack_dir))

    #Loop through every folder of the form l2dis_0.0x
    for l2_dir in l2_dir_list:
        images      = [f for f in os.listdir(l2_dir) if os.path.isfile(os.path.join(l2_dir, f))]
        l2_distance = os.path.basename(os.path.normpath(l2_dir))
        output_dir  =  denoise_base_dir + jpeg.__name__ + '_' + str(approx_level) + '/' + attack_name + '/' + l2_distance + '/'   

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for image in images:
            im1 = Image.open(os.path.join(l2_dir, image))
            filename, file_extension = os.path.splitext(image)
            jpeg_filename = filename + ".jpg"
            IMAGE_10 = os.path.join(output_dir, jpeg_filename)
            im1.save(IMAGE_10,"JPEG", quality=approx_level)
            im10 = Image.open(IMAGE_10)
