import io
import json

import numpy as np
from PIL import Image
import requests
from numpy import asarray


SERVER_URL = 'https://tensorflow-linear-model-0khx.onrender.com/v1/models/flowers-model:predict'

score = 0


def main():
  img = Image.open('rose.jpg')
  #img = Image.open('dog.png')
  img = img.resize((64, 64))
  img_array = asarray(img)

  img = np.expand_dims(img_array, 0).tolist()
  predict_request = json.dumps({'instances': img })

  # Send few actual requests and report average latency.
  total_time = 0
  num_requests = 1
  index = 0
  for _ in range(num_requests):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()
    total_time += response.elapsed.total_seconds()
    prediction = response.json()['predictions']
#    score = float(sigmoid[0](prediction[0][0]))
    print(prediction[0]) 
    classes_labels = ['dandelion', 'sunflowers', 'daisy', 'tulips', 'roses']
    print(classes_labels)
    image_output_class = classes_labels[np.argmax(prediction[0])]
    print(np.argmax(prediction[0]))

    print("The predicted class is", image_output_class)

#    print(response.json())
#    print ('sigmoid ', sigmoid[0](prediction[0][0]))




if __name__ == '__main__':
  main()
