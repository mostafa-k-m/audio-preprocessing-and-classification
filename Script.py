import librosa
import numpy as np
import pandas as pd


def data_loader(path, label):
    files = librosa.util.find_files(path)
    files = np.asarray(files)
    list = []
    for i in files:
        y, sr = librosa.load(i)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        list.append(mfcc)
        length = mfcc.shape[1]
    processed = []
    for i in list:
        means = np.mean(i,axis =1).reshape(1,-1)
        processed.append(means)
    data = np.vstack(processed)
    return data, np.asarray([label for i in range(data.shape[0])])


import os
train_spk1_path = os.path.join(os.curdir, "data/train/spk_1/")
train_spk2_path = os.path.join(os.curdir, "data/train/spk_2/")
test_spk1_path = os.path.join(os.curdir, "data/test/spk_1/")
test_spk2_path = os.path.join(os.curdir, "data/test/spk_2/")

train_spk1, y_train_spk1 = data_loader(train_spk1_path,0)
train_spk2, y_train_spk2 = data_loader(train_spk2_path,1)
test_spk1, y_test_spk1 = data_loader(test_spk1_path,0)
test_spk2, y_test_spk2 = data_loader(test_spk2_path,1)


X_train = np.vstack([train_spk1, train_spk2])
X_test = np.vstack([test_spk1, test_spk2])

y_train = np.hstack([y_train_spk1, y_train_spk2])
y_test = np.hstack([y_test_spk1, y_test_spk2])

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

X_train.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


grid_params = {
    'n_neighbors': [2, 3, 4],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
