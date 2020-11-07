import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from CNN_models.vgg import VGG16
#from CNN_models.ResNet18 import ResNet18
from models.CNN_models.lenet import build_LeNet #LeNet
from models.CNN_models.MiniVGGNet import MiniVGGNetModel
#from CNN_models.name_of_file_import name_of_class

def get_network(CNN_name, dense_layers) -> None:
    #create dense network dynamically based on input
    layer_list=[keras.Input(shape=(4,))]
    for i in range(len(dense_layers)):
        name = "layer" + str(i+1)
        layer_list+=[layers.Dense(dense_layers[i], activation="relu", name=name)] #final output layer, no activation for next layer

    dense_model = keras.Sequential(layer_list)
    print(dense_model.summary())
    #select the CNN network

    if CNN_name == 'LeNet':
        # CNN_model = LeNet()
        CNN_model = build_LeNet()
    elif CNN_name == 'MiniVGG':
        CNN_model = MiniVGGNetModel()
    elif CNN_name == 'VGG16':
        CNN_model = VGG16()
    elif CNN_name == 'ResNet':
        CNN_model = ResNet18()
    else:
        CNN_model = None
    print('True')
    #CNN_model.build((None, 32, 32, 3))
    # print(CNN_model.model().summary())
    print(CNN_model.summary())
    return dense_model, CNN_model

# get_network('LeNet',[5,4,4])
#get_network('MiniVGG',[5,4,4])
