## Setup ##
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# This code is showing the result of CAE
# to see the difference between input image and CAE-reconstructed ouput image

if __name__ == '__main__':
    path_dir = 'your image dir path'
    encoder_path = 'your CAE encoder path'
    decoder_path = 'your CAE decoder path'

    file_name = []
    file_list = []
    for (root, directories, files) in os.walk(path_dir):
        for file in files:
            temp = file.split('.')
            file_name.append(temp[0])
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    data_num = len(file_list)
    img_stacking = []
    count = 0

    for img in file_list:
        np_img = np.asarray(Image.open(img)) / 255.
        img_stacking.append(np_img)
        count += 1
        print(str(count) + " / " + str(data_num) + " Stack Finished ...")

    img_stacking = np.expand_dims(img_stacking, -1)

    if (os.path.exists(encoder_path)):
        encoder = keras.models.load_model(encoder_path, compile=False)
        # encoder.summary()
        print("Encoder model exist & loaded ...")

    if (os.path.exists(decoder_path)):
        decoder = keras.models.load_model(decoder_path, compile=False)
        # decoder.summary()
        print("Decoder model exist & loaded ...")

    # Latent vector code
    latent_z = encoder.predict(img_stacking)
    # print(latent_z)
    recon_img = decoder.predict(latent_z) # This recon_img is the result of CAE
    for i in range(data_num):
        plt.imshow(np.reshape(recon_img[i], (128, 128, 1)), cmap='gray')
        plt.axis('off')
        plt.savefig('your fig save dir', bbox_inches='tight', pad_inches=0) # This will save the output image with no padding
        plt.show()
