import os
import csv
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from scipy.stats import skew


def Load_File_Name(passed_dir):
    file_list = []
    for (root, directories, files) in os.walk(passed_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    return file_list


def Stackin_Data(img_file_list, CP_file_list):
    img_file_num = len(img_file_list)
    # CP_file_num = len(CP_file_list)
    img_stack = []
    CP_stack = []
    count = 0

    for img, CP in zip(img_file_list, CP_file_list):
        # Stack image
        np_img = np.asarray(Image.open(img)) / 255.
        img_stack.append(np_img)

        # Stack collapse percentage csv
        reader = list(csv.reader(open(CP)))
        now_b = np.squeeze(reader)
        now_b = [float(now_b)]
        now_b = np.transpose(np.reshape(np.asarray(now_b), -1))
        CP_stack.append(now_b)
        count += 1

        print(str(count) + " / " + str(img_file_num) + " Image & CP Stack Finished ...")

        # Debugging code
        # if count == 1000:
        #     break

    return img_stack, CP_stack


if __name__ == '__main__':

    latent_dim = 10

    ## Stacking a dataset ##
    img_path_dir = 'your image path'
    img_file_list = Load_File_Name(img_path_dir)
    img_data_num = len(img_file_list)

    CP_path_dir = 'your regression path'
    CP_file_list = Load_File_Name(CP_path_dir)
    CP_data_num = len(CP_file_list)

    if img_data_num == CP_data_num:
        img_stacking, CP_stacking = Stackin_Data(img_file_list, CP_file_list)

    else:
        sys.stderr.write("Data numbers are not equal! Try again ...")
        exit(1)

    encoder_path = 'your encoder path.h5'
    if (os.path.exists(encoder_path)):
        encoder = tf.keras.models.load_model(encoder_path, compile=False)
        # encoder.summary()
        print("Encoder model exist & loaded ...")

    MLP_path = 'your regression model path.h5'
    if (os.path.exists(MLP_path)):
        MLP = tf.keras.models.load_model(MLP_path, compile=True)
        # encoder.summary()
        print("MLP model exist & loaded ...")

    img_stacking = np.expand_dims(img_stacking, -1)
    CP_stacking = np.squeeze(CP_stacking)

    CAENR_input = Input(shape=(128, 128, 1), name='CAE_input')
    CAENR_encoder_output = encoder(CAENR_input)
    CAEMLP_output = MLP(CAENR_encoder_output)
    CAENR = Model(CAENR_input, CAENR_output, name='CAE')
    CAENR.summary()

    # CAEMLP.compile(optimizer='adam', loss='mse', metric=['mse'])
    # CAEMLP_test_scores = CAEMLP.evaluate(img_stacking, CP_stacking, verbose=1)
    # print("Test Loss : ", CAEMLP_test_scores[0])

    CP_predict = CAENR.predict(img_stacking)
    CP_predict = np.squeeze(CP_predict)
    diff = []

    for CAENR_result, CP in zip(CP_predict, CP_stacking):
        diff.append(abs(CP-CAENR_result))

    # print(diff)
    for CP, now_CP_result, now_diff, now_file in zip(CP_stacking, CP_predict, diff, img_file_list):
        print(now_file + ' : ' + str(CP) + ' / ' + str(now_CP_result) + ' / ' + str(now_diff))
    print(np.mean(diff))
    print(np.std(diff))
    print(skew(diff))

    
    # You can save CAENR model using save method
