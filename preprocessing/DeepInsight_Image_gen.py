#!/bin/python
from pyDeepInsight import ImageTransformer
from pyDeepInsight.utils import Norm2Scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from pathlib import Path
import umap.umap_ as umap

#directory where microbial matrices are formed
data_directory = Path("/scratch/general/vast/u6049572/Micro_matrices/antagonism")

#directory where you want the output to put saved
data_deepInsight = Path("/scratch/general/vast/u6049572/DeepInsight/antagonism")

tf.config.threading.set_inter_op_parallelism_threads(12)

def preprocess_image(image):
    # Normalize pixel values to [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

for file in os.listdir(data_directory):
    if file.endswith(".csv"):  # Ensure file is a CSV file
        print("Processing file:", file)
        try:
            with open(os.path.join(data_directory, file)) as f:
                com_file = f
                com = pd.read_csv(com_file, sep=",")
                X = com.iloc[:, :].values

                num_species = com.columns.values
                if num_species.shape == 3000:
                    #UMAP preserves local and global features, making it better than TSNE
                    reducer = umap.UMAP(
                        n_components=2,
                        #min_dist=0.8,
                        metric='cosine',
                        n_jobs=1,
                        random_state = 66,
                        )

                    pixel_size = (224,224)
                    it = ImageTransformer(
                        feature_extractor=reducer,
                        pixels=pixel_size)

                    image = it.fit(X, plot=False)
                    X_train_img = it.transform(X)

                    X_train_tensor = tf.stack([preprocess_image(img) for img in X_train_img])
                    X_array = X_train_tensor.numpy()
                    X_array = X_array[1,:,:,:]
                    print(X_array.shape)

                    #np.save(os.path.join(data_deepInsight, f'{file}_DeepInsight'), X_array)
                else:
                     #UMAP preserves local and global features, making it better than TSNE
                    reducer = umap.UMAP(
                        n_components=2,
                        #min_dist=0.8,
                        metric='cosine',
                        n_jobs=1,
                        random_state = 66
                        )

                    pixel_size = (224,224)
                    it = ImageTransformer(
                        feature_extractor=reducer,
                        pixels=pixel_size)

                    image = it.fit(X, plot=True)
                    X_train_img = it.transform(X)

                    X_train_tensor = tf.stack([preprocess_image(img) for img in X_train_img])
                    X_array = X_train_tensor.numpy()
                    X_array = X_array[1,:,:,:]
                    print(X_array.shape)

                    #np.save(os.path.join(data_deepInsight, f'{file}_DeepInsight'), X_array)
                print("Processing completed for file:", file)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
    else:
	    print(f"Ignoring non-CSV file: {file}")

