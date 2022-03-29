import os
import csv
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split


def sampling(mu_log_var):
    mu, log_var = mu_log_var
    sampling_epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tf.keras.backend.exp(log_var/2) * sampling_epsilon
    return random_sample


def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000 # This value is used only for vizual comfort
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=[1, 2, 3])
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=[1, 2, 3])
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)
        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss


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


def MLP_model(z_dim):
    MLP_model = tf.keras.Sequential(name='MLP_Model')
    MLP_model.add(tf.keras.layers.Dense(64, input_shape=(z_dim,)))
    MLP_model.add(tf.keras.layers.Dropout(0.5))
    MLP_model.add(tf.keras.layers.BatchNormalization())
    MLP_model.add(tf.keras.layers.ReLU())
    MLP_model.add(tf.keras.layers.Dense(64))
    MLP_model.add(tf.keras.layers.Dropout(0.5))
    MLP_model.add(tf.keras.layers.BatchNormalization())
    MLP_model.add(tf.keras.layers.ReLU())
    MLP_model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    # Since this is linear model, metrics=['acc'] is not necessary ('acc' is for classification)
    # https://stackoverflow.com/questions/45632549/why-is-the-accuracy-for-my-keras-model-always-0-when-training
    MLP_model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    MLP_model.summary()
    return MLP_model


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

    # Expand dims to fit the input of encoder
    img_stacking = np.expand_dims(img_stacking, -1)
    encoder_path = 'encoder_'+str(latent_dim)+'_'+image_method+'.h5'

    # Load encoder of CAE
    if (os.path.exists(encoder_path)):
        encoder = tf.keras.models.load_model(encoder_path, compile=False)
        # encoder.summary()
        print("Encoder model exist & loaded ...")

    else:
        print("There is no file! Check " + encoder_path + ' ...')

    encoder_result_stacking = encoder.predict(img_stacking)

    # Reshape dimension to fit on model
    # encoder_result_stacking = encoder_result_stacking[-1]
    # encoder_result_stacking = np.expand_dims(encoder_result_stacking, 1)
    CP_stacking = np.expand_dims(CP_stacking, 1)

    # print(np.shape(encoder_result_stacking)) # → Result : (n, 10)
    # print(np.shape(CP_stacking)) # → Result : (n, 1, 1)

    x_train, x_test, y_train, y_test = train_test_split(encoder_result_stacking, CP_stacking, shuffle=True, test_size=0.15)
    print("Data split Finished ...")

    stage2_model = MLP_model(z_dim=latent_dim)
    history = stage2_model.fit(x_train, y_train, validation_split=0.15, epochs=100, batch_size=50, verbose=1, shuffle=True)
    test_scores = stage2_model.evaluate(x_test, y_test, verbose=0, batch_size=10)
    # print(stage2_model.metrics_names) # → ['loss', 'mae']
    print("Test Loss : ", test_scores[0])
    # print("Test Accuracy : ", test_scores[-1])

    # Show plot of loss and accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('NR Model Loss (Z → Collapse Percentage)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()
    # Why validation loss is lower tha training loss?
    # https://stats.stackexchange.com/questions/287056/strange-training-loss-and-validation-loss-patterns

    stage2_model_path = 'your regression path.h5'
    stage2_model.save(stage2_model_path)
