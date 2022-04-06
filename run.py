import os
import glob
import argparse
from tkinter import image_names
from PIL import Image
from cv2 import ocl_Device
import numpy as np


def to_img(_arr):
  img = _arr[...,0]
  img /= np.max(img)
  img *= 255
  return img.astype(np.uint8)

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from tensorflow.keras.layers import Layer, InputSpec
from utils import predict, load_images, display_images
from matplotlib import offsetbox, pyplot as plt
import numpy as np

model_name = 'nyu'

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default=model_name+'.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='mzs_input/*.tif', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)
#model.summary()

print('\nModel loaded ({0}).'.format(args.model))

# Input images
img_names = glob.glob(args.input)
for imgn in img_names:
    path, ext = os.path.splitext(imgn)
    folder, name = os.path.basename(path)
    print(folder, name)
    continue
    name = os.path.splitext(imgn)[0] + '/' + model_name + '_D'
    print(name)
    inputs = load_images([imgn] )
    outputs = predict(model, inputs)
    with open(name+'.npy', 'wb') as f:
        np.save(f, outputs)
    Image.fromarray(to_img(outputs[0]), mode="L").save(name+'.tif')


