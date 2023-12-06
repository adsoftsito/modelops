import io
import json

import numpy as np
from PIL import Image
import requests
from numpy import asarray


sigmoid = (
  lambda x:1 / (1 + np.exp(-x)),
  lambda x:x * (1 - x)
  )


SERVER_URL = 'https://catdog-model-service-adsoftsito.cloud.okteto.net/v1/models/catdog-model:predict'

score = 0


def main():
  img = Image.open('cat.jpeg')
  #img = Image.open('dog.png')
  img = img.resize((180,180))
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
    score = float(sigmoid[0](prediction[0][0]))

    print(response.json())
    print ('sigmoid ', sigmoid[0](prediction[0][0]))

    print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")


if __name__ == '__main__':
  main()
