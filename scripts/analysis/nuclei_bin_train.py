from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras import optimizers
from keras import losses
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import EarlyStopping

import os
from glob import glob 
import datetime
import argparse
import logging

#from skimage.transform import resize 
#import umap
#import seaborn as sns 
import pandas as pd
import collections
import tifffile as tif
import PIL

import numpy as np
import random 
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import math

#import matplotlib.pyplot as plt
#plt.gray()
#import seaborn as sns
#sns.set_style("ticks")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# test if using gpu
from tensorflow import Session, ConfigProto
from tensorflow.test import gpu_device_name
sess = Session(config=ConfigProto(log_device_placement=True))

# test if using gpu
if gpu_device_name():
    logging.info('Default GPU Device: {}'.format(gpu_device_name()))
else:
    logging.info("Please install GPU version of TF")

parser = argparse.ArgumentParser(description='Set paths')
parser.add_argument('--dir_path', type=str, required=True,
                    help='the directory path where analysis is located')
parser.add_argument('--path_manual_txt', type=str, required=True,
                    help='the path to the manual count.txt file')
args = parser.parse_args()
dir_path = args.dir_path
path_manual = args.path_manual

# Set parameters:
staging_path = os.path.join(dir_path, 'nuclei_count')
tiles_dir = 'tiles_normed_embryo_wise_all_embryos'
tiles_path = os.path.join(staging_path, tiles_dir)
masked_cropped_20slices_path = os.path.join(staging_path, 'masked_cropped_20slices_all_embryos')

n_bins = 6
n_batches = 9
INPUT_SHAPE = (64,64,1)
BATCH_SIZE = 1024
N_EPOCHS = 20
r_dropouts = 0.1
n_conv_cnls = 4
verbose=1 # display training progress


def get_name_and_bin_training(path_manual):
    
    with open(path_manual, 'r') as f:
        lines = f.read().split('\n')
        
    names = [l.split(',')[0] for l in lines if l.split(',')[1]!='-1']
    bins = [int(l.split(',')[1]) for l in lines if l.split(',')[1]!='-1']   
    
    return names, bins


def plot_n_examples_per_bin(bins):

    bin_labels, bin_count = np.unique(bins, return_counts=True)

    l_n = ['1-4','5-30','31-99','100-149','150-557','558']
    clrs = ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A","#19D3F3"]

    fig, ax = plt.subplots(figsize=(15,6))
    ax.bar(l_n, bin_count, width=0.8, color=clrs)

    for b in bin_labels:
        ax.annotate(bin_count[b], xy=(b, bin_count[b]), xytext=(b-0.1, bin_count[b]+0.9), fontsize=20)

    ax.set_xlabel('Bins (#nuclei)')
    ax.set_ylabel(f'#embryos, n={sum(bin_count)}')
    ax.set_ylim(0,90)

    ax.hlines(min(bin_count), -0.5, 5.5, linestyle='dashed', colors='blue')

    # fig = plt.gcf()
    # fig.savefig("thesis_manual_annotations_stage.pdf", bbox_inches="tight")


def get_dataframes(dir_path, names, bins):
    df_all = pd.read_csv(os.path.join(dir_path, 'embryos_csv', 'embryos.csv'))
    df_all = df_all[["cropped_image_file", "cropped_mask_file", "DAPI channel", "GFP channel"]]
    
    df_labeled = df_all[df_all.cropped_image_file.isin(names)]
    df_labeled = df_labeled.reset_index(drop=True)
    
    for i in df_labeled.index:
        df_labeled.loc[i,'binn'] = bins[names.index(df_labeled.loc[i,"cropped_image_file"])]
    
    return df_all, df_labeled


def get_embryos_with_no_tiles(df_labeled, tiles_path):
    
    missing = []
    
    for i in df_labeled.index:
        name = f'{df_labeled.at[i,"cropped_image_file"][:-4]}_tiles.tif'
        if not os.path.exists(os.path.join(tiles_path , name)):
            missing.append(name)
            
    logging.info(f'embryos with missing tiles file: {(os.linesep).join(missing)}')
    return missing


def add_n_tile_col_to_df(df, tiles_path):
    
    for i in df.index:
        im = PIL.Image.open(os.path.join(tiles_path,
                                         f'{df.loc[i,"cropped_image_file"][:-4]}_tiles.tif'))
        df.loc[i,"n_tiles"] = im.n_frames
                                         
    return df


def get_usable_tile_count(df, n_bins, n_per_bin):

    nth_largest_ntile_per_bin = []
    for i in range(n_bins):
        nth_largest_ntile_per_bin.append(df[df["binn"]==i].
                                            n_tiles.nlargest(n_per_bin).iloc[-1])

    usable_tile_count = int(min(nth_largest_ntile_per_bin))
    return usable_tile_count


def get_training_df(df, usable_tile_count, n_per_bin):
    
    df = df[df.n_tiles>=usable_tile_count]
    df = df.groupby('binn').head(n_per_bin)
    
    df = df.sort_values(by=['binn'])
    df = df.reset_index(drop=True)
    
    return df


def check_df_n_per_bin(df, n_bins):
    
    for i in range(n_bins):
        logging.info(f'bin {i} # examples: {df[df.binn==i].shape[0]}')
        
    logging.info(f'df min number of tiles: {df.n_tiles.min()}')


def get_list_training_tiles(df, tiles_path, usable_tile_count):
    
    all_tiles=[]
    for i in df.index:
        embryo_tiles = tif.imread(os.path.join(tiles_path, f'{df.loc[i,"cropped_image_file"][:-4]}_tiles.tif'))
        all_tiles.append(embryo_tiles[:usable_tile_count])
        
        if (i + 1) % 20 == 0:
            logging.info(f'Get training tiles: Processing iteration {i + 1} out of {df.shape[0]}')
                                            
    return all_tiles


def show_example_embryos(indxs, embryos_names, n_bins, ims_per_bin, slice_num=10):
    chosen_embryos_names = [name for i,name in enumerate(embryos_names) if i in indxs]
    
    embryos_ims = [tif.imread(os.path.join(masked_cropped_20slices_path, name))
                   for name in chosen_embryos_names]
    
    fig, axs = plt.subplots(ims_per_bin, n_bins, figsize=(20,20*ims_per_bin/n_bins))

    for i in range(n_bins):
        for j in range(ims_per_bin):
            idx = (i*ims_per_bin)+j
            axs[j,i].imshow(embryos_ims[idx][slice_num])
            axs[j,i].axis('off')
            if j%ims_per_bin==0:
                axs[j,i].text(10,40, f'class {int(idx/ims_per_bin)}', color='r', fontsize=30)
            axs[j,i].text(30,90, chosen_embryos_names[idx].split('_c')[0], color='g', fontsize=15)


def show_example_tiles(all_tiles, n_bins, n_per_bin, seed_im=42, beg_tile=0):

    fig, ax = plt.subplots(3, n_bins, figsize=(13,5))

    for row in ax:
        for ic,c in enumerate(row):
            random.seed(seed_im)
            im = random.choice(all_tiles[ic*n_per_bin:(ic+1)*n_per_bin])
            
            c.imshow(im[beg_tile,:,:,0])
            c.axis('off')
            c.text(5,10,f'class{ic}',color='y',fontsize=20, weight="bold")
        beg_tile+=5


# Function - prep for NN


def preprocess(embryos_tiles, embryos_labels, val_size=5, val_batch=0):
    
    ## define (embryos) split indexes - train & validation
    ## This is for n val embryos
    val_idxs = [(val_batch*val_size)+i for i in range(len(embryos_tiles)) if i%n_per_bin<val_size]
    train_idxs = [i for i in range(len(all_tiles)) if i not in val_idxs]
    
    # Split
    train_embryos_tiles = [embryos_tiles[i] for i in train_idxs]
    val_embryos_tiles = [embryos_tiles[i] for i in val_idxs]
    train_embryos_labels = [embryos_labels[i] for i in train_idxs]
    val_embryos_labels = [embryos_labels[i] for i in val_idxs]
    
    # Flatten tile arrays
    train_data = np.concatenate(train_embryos_tiles, axis=0)
    val_data = np.concatenate(val_embryos_tiles, axis=0)
    
    # create labels arrays
    if n_tiles:
        train_labels_lists = [[l]*n_tiles for i,l in enumerate(train_embryos_labels)]
        val_labels_lists = [[l]*n_tiles for i,l in enumerate(val_embryos_labels)]
    else:
        train_labels_lists = [[l]*n_tiles[train_idxs[i]] for i,l in enumerate(train_embryos_labels)]
        val_labels_lists = [[l]*n_tiles[val_idxs[i]] for i,l in enumerate(val_embryos_labels)]
    
    train_labels = np.concatenate(train_labels_lists)
    val_labels = np.concatenate(val_labels_lists)
    
    ## Do the same to the embryo index for the validation set
    ## so we can do stats - match the embryo index with the tile after training
    if n_tiles:
        val_embryo_idx_list = [[i]*n_tiles for i in val_idxs]
    else:
        val_embryo_idx_list = [[i]*n_tiles[i] for i in val_idxs]
    val_embryo_idx = np.concatenate(val_embryo_idx_list)
    
    # shuffel
    train_data, train_labels = shuffle(train_data, train_labels, random_state=42)
    val_data, val_labels = shuffle(val_data, val_labels, random_state=42)
    
    val_embryo_idx = shuffle(val_embryo_idx, random_state=42)
    
    ### Labels reshape to one hot (for softmax)
    train_Y = to_categorical(train_labels)
    val_Y = to_categorical(val_labels)

    return train_data, train_Y, val_data, val_Y, val_embryo_idx, val_idxs, val_embryos_labels


# Function - define NN


def encoder(input_img):

    x = Conv2D(n_conv_cnls, kernel_size=3, activation="relu", strides=2, padding="same")(input_img)
    x = BatchNormalization()(x)
    x = Dropout(r_dropouts)(x)
    
    x = Conv2D(n_conv_cnls*2, kernel_size=3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(r_dropouts)(x)
    
    x = Conv2D(n_conv_cnls*4, kernel_size=3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(r_dropouts)(x)
    
    x = Conv2D(n_conv_cnls*8, kernel_size=3, activation="relu", strides=2, padding="same")(x)
    i_o = BatchNormalization()(x)
    
    return i_o


def decoder(d_i): 
    #d_i = Input(encoder.output_shape[1:], name='decoder_input')

    x    = Conv2DTranspose(n_conv_cnls*8, kernel_size=3, strides=2, padding='same', activation='relu')(d_i)
    x    = BatchNormalization()(x)
    x = Dropout(r_dropouts)(x)
    
    x    = Conv2DTranspose(n_conv_cnls*4, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x    = BatchNormalization()(x)
    x = Dropout(r_dropouts)(x)
    
    x    = Conv2DTranspose(n_conv_cnls*2, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x    = BatchNormalization()(x)
    x = Dropout(r_dropouts)(x)
    
    x    = Conv2DTranspose(n_conv_cnls*1, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x    = BatchNormalization()(x)
    x = Dropout(r_dropouts)(x)

    o     = Conv2DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')(x)
    
    return o


def fully_connected(encod):
    flat = Flatten()(encod)
    den = Dense(128, activation='relu')(flat)
    out = Dense(n_bins, activation='softmax')(den)
    return out


# Function - NN loss metric


def plot_training_summary(history, is_ae=True):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if is_ae:
        plt.ylim((0, 0.01))
    else:
        plt.ylim((0.5, 2.5))
    plt.show()


# Function - Def run NN:


def train_autoencoder_then_classifier(autoencoder, classifier, n_epochs_ae=20, n_epochs_cla=15, 
                                      lr_ae=0.01, lr_cla=0.01):
    
    # AE
    for l1,l2 in zip(autoencoder.layers[:12],classifier.layers[:12]):
        l1.set_weights(l2.get_weights())
    
    es = EarlyStopping(monitor='loss',
                              min_delta=0,
                              patience=3,
                              verbose=0, mode='auto',restore_best_weights=True)
    
    autoencoder.compile(loss='mean_squared_error', optimizer =optimizers.Adam(lr=lr_ae))

    history_ae = autoencoder.fit(train_data, train_data, 
                          batch_size=BATCH_SIZE, epochs=n_epochs_ae, 
                              callbacks=[es], validation_data=(val_data, val_data))
    
    plot_training_summary(history_ae)
    
    # Classifier
    for l1,l2 in zip(classifier.layers[:12],autoencoder.layers[:12]):
        l1.set_weights(l2.get_weights())
        
    #classifier.save_weights(f'classifier_{str_time}.h5')
    classifier.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(lr=lr_cla))
    
    history_cl = classifier.fit(train_data, train_Y, 
                                batch_size=BATCH_SIZE,
                                epochs=n_epochs_cla,
                                verbose=1,
                                 callbacks=[es],
                                validation_data=(val_data, val_Y))
    
    str_time = '{0:%y%m%d%H%M%S}'.format(datetime.datetime.now())
    classifier.save_weights(f'classifier_{str_time}.h5')
    
    # Loss metrics:
    plot_training_summary(history_cl, is_ae=False)
    
    return classifier, history_ae, history_cl


if __name__ == "__main__":

    names, bins = get_name_and_bin_training(path_manual)
    #plot_n_examples_per_bin(bins)

    df_all, df_labeled = get_dataframes(dir_path, names, bins)

    logging.info(f'Number of labeled embryos: {len(names)}')
    logging.info(f'Number of labeled embryos in embryos.csv: {len(set(df_labeled["cropped_image_file"].tolist()))}')
    logging.info(f'unique bin values: {df_labeled["binn"].unique()}')

    get_embryos_with_no_tiles(df_labeled, tiles_path)

    n_per_bin = df_labeled["binn"].value_counts().min()

    df_labeled = add_n_tile_col_to_df(df_labeled, tiles_path)

    n_tiles = get_usable_tile_count(df_labeled, n_bins, n_per_bin)

    df = get_training_df(df_labeled, n_tiles, n_per_bin)
    check_df_n_per_bin(df, n_bins)

    df.to_csv(os.path.join(staging_path, 'training_data.csv'), index=False)

    all_tiles = get_list_training_tiles(df, tiles_path, n_tiles)

    logging.info(f'# training examples: {len(all_tiles)}')
    logging.info(f'number of tiles example training: {all_tiles[3].shape}')

    # list of list with the label repeated for all tiles of the embryos
    tiles_labels = [[int(df.loc[i,"binn"])]*all_tiles[i].shape[0] for i in df.index]

    n_images_per_bin = 3

    # show_example_embryos([i for i in range(df.shape[0]) if i%n_per_bin<n_images_per_bin], 
    #                      df.cropped_image_file.tolist(), n_bins, 
    #                      n_images_per_bin)

    #show_example_tiles(all_tiles, n_bins, n_per_bin)


    classifier_and_histories = []

    for i in range(n_batches):
        
        logging.info(f'batch number {i}')
        
        train_data, train_Y, val_data, val_Y, val_embryo_idx, val_idxs, val_embryos_labels = preprocess(all_tiles, df.binn.to_list(), val_size=math.floor(n_per_bin/n_batches), val_batch=i)
        
        input_img = Input(shape = INPUT_SHAPE)
        autoencoder = Model(input_img, decoder(encoder(input_img)))
        autoencoder.compile(loss='mean_squared_error', optimizer = 'adam')

        encode = encoder(input_img)
        classifier = Model(input_img, fully_connected(encode))

        classifier_and_histories.append(train_autoencoder_then_classifier(autoencoder, classifier))