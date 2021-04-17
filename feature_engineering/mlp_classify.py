import numpy as np
from keras.utils import to_categorical
from keras import models
from sklearn.preprocessing import StandardScaler
from keras import layers
import tqdm
import pickle
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from keras.optimizers import SGD
import pandas as pd
import matplotlib.pyplot as plt


batch_size = 512
epochs = 1000

with open('data/x_train.pkl', 'rb') as f:
    x_train = pickle.load(f)
with open('data/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('data/x_test.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('data/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

x_train_numpy = np.array(x_train)
x_test_numpy = np.array(x_test)
y_train_numpy = y_train['label'].values
y_test_numpy = y_test['label'].values

x_train_numpy = x_train_numpy[y_train_numpy != 3]
x_test_numpy = x_test_numpy[y_test_numpy != 3]
y_train_numpy = y_train_numpy[y_train_numpy != 3]
y_test_numpy = y_test_numpy[y_test_numpy != 3]

num_features = x_train_numpy.shape[1]
num_classes = 3

scaler = StandardScaler()
scaler.fit(x_train_numpy)
x_train_numpy = scaler.transform(x_train_numpy)
x_test_numpy = scaler.transform(x_test_numpy)

y_train_one_hot = to_categorical(y_train_numpy, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test_numpy, num_classes=num_classes)

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(num_features,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# model.fit(x_train_numpy, y_train_one_hot, epochs=9, batch_size=512, shuffle=True)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, verbose=2, mode='auto', restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath='check_point', monitor='val_accuracy', verbose=2,
                             save_best_only=True, mode='max', save_freq='epoch')
callbacks_list = [checkpoint, early_stopping]
# callbacks_list = [checkpoint]

rmsprop = keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(x_train_numpy, y_train_one_hot, epochs=epochs,
#                     batch_size=batch_size, validation_split=0.2, shuffle=True, callbacks=callbacks_list)
history = model.fit(x_train_numpy, y_train_one_hot, epochs=epochs, batch_size=batch_size,
                    validation_data=(x_test_numpy, y_test_one_hot), shuffle=True, callbacks=callbacks_list)

y_pred = model.predict(x_test_numpy)
y_pred2 = y_pred.argmax(axis=1)
print(accuracy_score(y_test_numpy, y_pred2))
aaa = confusion_matrix(y_test_numpy, y_pred2)
bbb = (aaa.T / aaa.sum(axis=1)).T
print(pd.DataFrame(bbb))

yy_test_numpy = keras.utils.to_categorical(y_test_numpy, 3)
plt.figure(figsize=(20, 6))
for i in range(3):
    y_test_x = [j[i] for j in yy_test_numpy]
    y_predict_x = [j[i] for j in y_pred]
    fpr, tpr, thresholds = metrics.roc_curve(y_test_x, y_predict_x, pos_label=1)
    plt.subplot(1, 3, i + 1)
    plt.plot(fpr, tpr)
    plt.grid()
    plt.plot([0, 1], [0, 1])
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    auc = np.trapz(tpr, fpr)
    print('AUC:', auc)
    plt.title('label' + str(i) + ', AUC:' + str(auc))
plt.show()


################################
x_train_numpy = np.array(x_train)
x_test_numpy = np.array(x_test)
y_train_numpy = y_train['label'].values
y_test_numpy = y_test['label'].values

train_mask = (y_train_numpy != 3) & (y_train_numpy != 2)
test_mask = (y_test_numpy != 3) & (y_test_numpy != 2)

x_train_numpy = x_train_numpy[train_mask]
x_test_numpy = x_test_numpy[test_mask]
y_train_numpy = y_train_numpy[train_mask]
y_test_numpy = y_test_numpy[test_mask]
num_features = x_train_numpy.shape[1]

scaler = StandardScaler()
scaler.fit(x_train_numpy)
x_train_numpy = scaler.transform(x_train_numpy)
x_test_numpy = scaler.transform(x_test_numpy)

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(num_features,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, verbose=2, mode='auto', restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath='check_point', monitor='val_accuracy', verbose=2,
                             save_best_only=True, mode='max', save_freq='epoch')
callbacks_list = [checkpoint, early_stopping]

rmsprop = keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

# history = model.fit(x_train_numpy, y_train_numpy, epochs=epochs,
#                     batch_size=batch_size, validation_split=0.2, shuffle=True, callbacks=callbacks_list)
history = model.fit(x_train_numpy, y_train_numpy, epochs=epochs, batch_size=batch_size,
                    validation_data=(x_test_numpy, y_test_numpy), shuffle=True, callbacks=callbacks_list)

y_pred = model.predict(x_test_numpy).flatten()
y_pred2 = (y_pred > 0.5).astype('int')
print(accuracy_score(y_test_numpy, y_pred2))
aaa = confusion_matrix(y_test_numpy, y_pred2)
bbb = (aaa.T / aaa.sum(axis=1)).T
print(pd.DataFrame(bbb))

plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = metrics.roc_curve(y_test_numpy, y_pred, pos_label=1)
plt.plot(fpr, tpr)
plt.grid()
plt.plot([0, 1], [0, 1])
plt.xlabel('FP')
plt.ylabel('TP')
plt.ylim([0, 1])
plt.xlim([0, 1])
auc = np.trapz(tpr, fpr)
print('AUC:', auc)
plt.title('AUC:' + str(auc))
plt.show()

