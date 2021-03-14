import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from models import get_network, create_concat_network

import numpy as np
import os,pandas as pd
import platform
import cv2

def tester(bathroom_image_path,frontal_image_path,bedroom_image_path,kitchen_image_path, bedrooms, bathrooms, postal_code, sqft):
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    MERGED_IMAGE_HEIGHT = IMAGE_HEIGHT*2
    MERGED_IMAGE_WIDTH = IMAGE_WIDTH*2

    img_list = [cv2.imread(bathroom_image_path), cv2.imread(frontal_image_path), cv2.imread(bedroom_image_path), cv2.imread(kitchen_image_path)]
    # first resize the 4 inputs to 32 x 32
    for i, image in enumerate(img_list):
        img_list[i] = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    merged_image = np.zeros((MERGED_IMAGE_HEIGHT, MERGED_IMAGE_WIDTH, 3), np.uint8)#img_list[0].dtype)
    if platform.system()=='Windows':
        merged_image[0:IMAGE_HEIGHT, 0: IMAGE_WIDTH] = img_list[0]
        merged_image[IMAGE_HEIGHT: MERGED_IMAGE_HEIGHT, 0: IMAGE_WIDTH] = img_list[1]
        merged_image[0:IMAGE_HEIGHT, IMAGE_WIDTH: MERGED_IMAGE_WIDTH] = img_list[2]
        merged_image[IMAGE_HEIGHT: MERGED_IMAGE_HEIGHT, IMAGE_WIDTH: MERGED_IMAGE_WIDTH] = img_list[3]
    else:
        merged_image[0:IMAGE_HEIGHT, 0: IMAGE_WIDTH] = img_list[2]
        merged_image[IMAGE_HEIGHT: MERGED_IMAGE_HEIGHT, 0: IMAGE_WIDTH] = img_list[3]
        merged_image[0:IMAGE_HEIGHT, IMAGE_WIDTH: MERGED_IMAGE_WIDTH] = img_list[0]
        merged_image[IMAGE_HEIGHT: MERGED_IMAGE_HEIGHT, IMAGE_WIDTH: MERGED_IMAGE_WIDTH] = img_list[1]


    df = pd.read_csv('CA.txt', delimiter = "\t")
    toronto_dataset = pd.read_excel('relevant_files'+os.sep+'TorontoDataset.xlsx')
    main_dataframe = df.loc[:,["T0A","54.766","-111.7174"]]
    main_dataframe = main_dataframe.rename(columns={"T0A": "Zip Code", "54.766": "Lat", "-111.7174":"Long"})
    temp_try = pd.merge(toronto_dataset,main_dataframe,how="inner")

    bedrooms_column = temp_try["Bedrooms"]
    normalized_bedroom_num = (bedrooms - bedrooms_column.min()) / (bedrooms_column.max() - bedrooms_column.min())
    bathrooms_column = temp_try['Bathrooms']
    normalized_bathroom_num = (bathrooms - bathrooms_column.min()) / (bathrooms_column.max() - bathrooms_column.min())
    sqft_column = temp_try['Sqft']
    normalized_sqft_num = (sqft - sqft_column.min()) / (sqft_column.max() - sqft_column.min())

    zipcode_column = temp_try["Zip Code"]
    matching_rows = temp_try[zipcode_column == postal_code]
    latitude = matching_rows['Lat'].values[0]
    longtitude = matching_rows['Long'].values[0]

    lat_column = temp_try['Lat']
    normalized_lat = (latitude - lat_column.min()) / (lat_column.max() - lat_column.min())
    long_column = temp_try['Long']
    normalized_long = (longtitude - long_column.min()) / (long_column.max() - long_column.min())

    max_price = temp_try['Price'].max()

    output = (normalized_bedroom_num, normalized_bathroom_num, normalized_sqft_num, normalized_lat, normalized_long)

    CNN_type = 'RegNet'

    Dense_NN, CNN = get_network(CNN_type, dense_layers=[8,4], \
    CNN_input_shape= [64, 64, 3], input_shape=5)

    Multi_Input = tf.keras.layers.concatenate([Dense_NN.output, CNN.output])

    #Not updated from 63 commit
    model = Model(inputs = [Dense_NN.input , CNN.input], outputs = create_concat_network(Multi_Input), name="combined")
    model.load_weights('relevant_files'+os.sep+'model_weights.h5')

    output = np.reshape(np.array(output),(1,5))
    merged_image = np.reshape(merged_image,(1,MERGED_IMAGE_WIDTH,MERGED_IMAGE_WIDTH,3))

    q = model.predict([output,merged_image])[0][0]
    final_prediction = q*max_price
    print(q)
    return output

if __name__=='__main__':
    bathroom_path = os.path.join(os.getcwd(), 'raw_dataset', '1_bathroom.jpg')
    frontal_path = os.path.join(os.getcwd(), 'raw_dataset', '1_frontal.jpg')
    bedroom_path = os.path.join(os.getcwd(), 'raw_dataset', '1_bedroom.jpg')
    kitchen_path = os.path.join(os.getcwd(), 'raw_dataset', '1_kitchen.jpg')

    tester(bathroom_path, frontal_path, bedroom_path, kitchen_path, 2, 3, 'M9P', 1100)
