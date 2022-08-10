# TFLite quantized inference example
#
# Based on:
# https://www.tensorflow.org/lite/performance/post_training_integer_quant
# https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor.QuantizationParams
import cv2
from cProfile import label
import numpy as np
import tensorflow as tf
import time 
import os
from tensorflow import keras
import tensorflow_datasets as tfds

'''
def evaluate_model(interpreter, ds, model_name):
  output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for i, (test_image, test_labels) in enumerate(ds):
    #print(test_image.shape, test_image[0].shape, test_labels.shape)
    
    
    for id in range(10):
      cv2.imshow('image',test_image[id].numpy())
      cv2.waitKey(0)
    

    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    print(output())
    digit = np.argmax(output(), axis=1)
    print(digit, test_labels)
    prediction_digits.append(digit)
    
    break

  print('\n')
  # Compare prediction results with ground truth labels to calculate accuracy.
  prediction_digits = np.array(prediction_digits)
  accuracy = (prediction_digits == test_labels.numpy()).mean()
  print(model_name, 'model test_accuracy:', accuracy*100, "%")
'''

def normalize_resize(image):
    #cv2.imshow('image',image)
    #cv2.waitKey(0)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    #image = tf.image.convert_image_dtype(image, tf.float32)
    for channel in range(3):
           image[:,:,channel] = (image[:,:,channel] - mean[channel]) / std[channel]
    image = cv2.resize(image, (32, 32))
    #cv2.imshow('image',image)
    #cv2.waitKey(0)
    #exit()
    return image

# Load cifar dataset
global test_labels
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
test_images = test_images / 255.0
test_images = np.array([normalize_resize(img) for img in test_images])



def evaluate_model(interpreter):
  global test_labels
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  prediction_digits = np.empty([0])

  for i in range(len(test_images)):
    
    # for batch 
    # test_image = test_images[i*10:i*10+10, : ,: ,: ].astype(np.float32)
    test_image = np.expand_dims(test_images[i], axis=0).astype(np.float32)
    
    #if i % 1000 == 0:
    #  print('Evaluated on {n} results so far.'.format(n=i))
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    #test_image = np.expand_dims(test_image, axis=0).astype(np.float32)

    '''
    for id in range(10):
      cv2.imshow('image',test_image[id])
      cv2.waitKey(0)
    '''
    #exit()
    
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output(), axis=1)
    prediction_digits = np.concatenate((prediction_digits, digit), axis=0)

  # Compare prediction results with ground truth labels to calculate accuracy.
  prediction_digits = np.array(prediction_digits)
  
  # for batch 
  #test_labels = np.array(test_labels[0:100])
  test_labels = np.array(test_labels)

  num_correct = 0
  for i in range(len(prediction_digits)):
      if prediction_digits[i] == test_labels[i]:
        num_correct += 1
  accuracy = num_correct/len(test_labels)*100
  print("ACC: ", accuracy, "%")
  return accuracy


def inference(model_path, model_name):
  # Load TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path=model_path)
  #interpreter.resize_tensor_input(0, [num_test_images, 3, 7, 7])

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
 
  #num_test_images = 10
  # Adjust graph input to handle batch tensor
  #interpreter.resize_tensor_input(input_details[0]['index'], [num_test_images, 3, 7, 7]) 

  # Adjust output #1 in graph to handle batch tensor
  #interpreter.resize_tensor_input(output_details[0]['index'], [num_test_images, 3, 7, 7]) 


  interpreter.allocate_tensors()  

  # Get input and output tensors.

  input_shape = input_details[0]['shape']

  dummy_input = np.array(np.random.random_sample(input_shape), dtype=np.float32)
  # warm up 
  for _ in range(0, 10):
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    interpreter.get_tensor(output_details[0]['index'])

  start = time.time()
  for _ in range(0, 100):
      interpreter.set_tensor(input_details[0]['index'], dummy_input)
      interpreter.invoke()
      interpreter.get_tensor(output_details[0]['index'])

  print(model_name, f' model cost time:{(time.time()- start)} sec')
  print(model_name, " model in Mb:", os.path.getsize(model_path) / float(2**20))

  evaluate_model(interpreter=interpreter)
  #evaluate_model(interpreter=interpreter, ds=test, model_name=model_name)

  return interpreter


if __name__ == '__main__':

  # Location of tflite model file (float32 or int8 quantized)
  quant_model_path = "../output/qat_model.tflite"
  float_model_path = "../output/float.tflite"
  origin_model_path = "../output/original.tflite"

  # Load CIFAR10 dataset.
  (ds_train, ds_val, ds_test), ds_info = tfds.load(
      'cifar10',
      split=['train[:90%]', 'train[90%:]', 'test'],
      as_supervised=True,
      with_info=True,
  )


  #train = train_ds.map(normalize_resize).cache().map(augment).shuffle(100).batch(64).repeat()
  #test = ds_test.map(normalize_resize).cache().batch(10)


  interpreter = inference(quant_model_path, "pruned + Quantized")
  #evaluate_model(interpreter=interpreter, ds=ds_test, model_name="pruned + Quantized")
  print()

  interpreter = inference(float_model_path, "pruned")
  #evaluate_model(interpreter=interpreter, ds=ds_test, model_name="pruned")
  print()

  #interpreter = inference(origin_model_path, "original")
  #evaluate_model(interpreter=interpreter, ds=ds_test, model_name="original")
  print()


'''

quant_test_accuracy = evaluate_model(interpreter)
float_test_accuracy = evaluate_model(interpreter2)
print('Quant model test_accuracy:', quant_test_accuracy)
print('Float model test accuracy:', float_test_accuracy)
'''