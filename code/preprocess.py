import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/fer2013.csv')

image_size = len(data.pixels[0].split(' '))
width = int(math.sqrt(image_size))
height = int(math.sqrt(image_size))
img_features = data['pixels'].apply(lambda x: np.array(x.split()).reshape(height, width, 1).astype('float32'))
img_features = np.stack(img_features, axis=0)
img_features = img_features / 255.0
img_labels = pd.get_dummies(data['emotion'])

X_train, X_valid, y_train, y_valid = train_test_split(img_features, img_labels,
                                                      shuffle=True, stratify=img_labels,
                                                      test_size=0.1, random_state=42)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
