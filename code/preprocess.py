import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('data/fer2013.csv', delimiter=',')
emotion_map = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'sadness',
    5: 'surprise',
    6: 'neutral'}

df_train = data[data["Usage"] == "Training"]
df_test_public = data[data["Usage"] == "PublicTest"]
df_test_private = data[data["Usage"] == "PrivateTest"]

image_size = len(df_train['pixels'].iloc[0].split(' '))
width = int(math.sqrt(image_size))
height = int(math.sqrt(image_size))
img_features = df_train['pixels'].apply(lambda x: np.array(x.split()).reshape(height, width, 1).astype('float32'))
img_features = np.stack(img_features, axis=0)
img_features = img_features / 255.0
img_labels = np.array(df_train['emotion'])

X_train, X_valid, y_train, y_valid = train_test_split(img_features, img_labels,
                                                      shuffle=True, stratify=img_labels,
                                                      test_size=0.3, random_state=42)
