# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 12:05:52 2020

@author: MSc. Maciej Rosoł
"""

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense,Flatten,Dropout,GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
import tensorflow as tf
import keras
from tensorflow.keras import regularizers
import timeit
import os
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
os.chdir(os.path.dirname(__file__))
    
#%% Loading training data
from matplotlib import image
os.chdir(os.path.dirname(__file__))
directory = os.getcwd()+'\\Linnaeus\\train'
images = []
labels = []
label = 0
for foldername in listdir(directory):
    for filename in listdir(directory+'\\'+foldername):
        img = image.imread(directory+'\\'+foldername+'\\'+filename)
        images.append(img)
        labels.append(label)
    label+=1
    
X_train = np.asarray(images)
y_train = np.asanyarray(labels).reshape(-1, 1)
#%% Initializing list and dictionaries

hist = []
time = []

def training(model, lr, X_train, y_train):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    mdl.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(X_train,y_train,
             epochs=5,
             batch_size=32,
             shuffle=True,
             verbose=True)
    return history, model
#%% Base ResNet50
base_model = ResNet50(include_top=False, input_shape=(256, 256, 3))
for j,layer in enumerate(base_model.layers):
    if j<len(base_model.layers)-4:
        layer.trainable=False
#%% Training the model
with tf.device('/gpu:0'):
       
    mdl = base_model.output
    mdl = GlobalAveragePooling2D()(mdl)
    mdl = Dense(1024,activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2))(mdl) 
    mdl = Dropout(0.05)(mdl)
    mdl = Dense(1024,activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2))(mdl)
    mdl = Dropout(0.05)(mdl)
    mdl = Dense(5,activation='softmax')(mdl)
    mdl = Model(inputs=base_model.inputs, outputs=mdl)
    
    time.append(timeit.default_timer())
            
    h, mdl = training(mdl, 0.001, X_train, y_train)
    hist.append(h)
    
    time.append(timeit.default_timer())
    
    h, mdl = training(mdl, 0.0001, X_train, y_train)
    hist.append(h)
    
    time.append(timeit.default_timer())
    
    h, mdl = training(mdl, 0.00001, X_train, y_train)
    hist.append(h)
    
    time.append(timeit.default_timer())
    
    h, mdl = training(mdl, 0.000001, X_train, y_train)
    hist.append(h)
    
    time.append(timeit.default_timer())
    
    mdl.save('final_model')

#%% Plotting history of training
hist_acc = []
hist_loss = []
for h in hist:
    hist_acc.append(h.history['accuracy'])
    hist_loss.append(h.history['loss'])
hist_acc=np.hstack(hist_acc)
hist_loss=np.hstack(hist_loss)
plt.figure()
plt.plot(hist_acc)
plt.legend(['train','val'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.figure()
plt.plot(hist_loss)
plt.legend(['train','val'])
plt.xlabel('Epochs')
plt.ylabel('Loss')

#%% Performance on training dataset
y_hat_train = mdl.predict(X_train)
y_pred_train = np.argmax(y_hat_train,axis=1)

acc_train = accuracy_score(y_train,y_pred_train)
cm_train = confusion_matrix(y_train,y_pred_train)

names = ['berry','bird','dog','flower','other']
plt.figure()
sns.heatmap(cm_train.astype(int), annot=True,fmt='d', cmap="YlGnBu", xticklabels=names,yticklabels=names)

#%% Loading test data
directory = os.getcwd()+'\\Linnaeus\\test'
images_test = []
labels_test = []
label = 0
for foldername in listdir(directory):
    for filename in listdir(directory+'\\'+foldername):
        img = image.imread(directory+'\\'+foldername+'\\'+filename)
        images_test.append(img)
        labels_test.append(label)
    label+=1
    
X_test = np.asarray(images_test)
y_test = np.asanyarray(labels_test).reshape(-1, 1)

#%% Prediction on test data
y_hat_test = mdl.predict(X_test)
y_pred_test = np.argmax(y_hat_test,axis=1)
acc_test = accuracy_score(y_test,y_pred_test)
cm_test = confusion_matrix(y_test,y_pred_test)

#%% Performance on the test data
precision, recall, F1, _ = precision_recall_fscore_support(y_test,y_pred_test)

specificity = []
accuracy = []

enc = OneHotEncoder(handle_unknown='ignore').fit(y_test.reshape(-1, 1))
enc_test = enc.transform(y_test.reshape(-1, 1)).toarray()
enc_pred = enc.transform(y_pred_test.reshape(-1, 1)).toarray()

tpfpfntp = {}
for i in range(5):
    y_true = enc_test[:,i]
    y_pred = enc_pred[:,i]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpfpfntp[names[i]] = [tn,fn,tp,fp]
    specificity.append(tn / (tn+fp))
    accuracy.append((tp+tn)/(tp+tn+fp+fn))
tpfpfntp = pd.DataFrame(tpfpfntp)
tpfpfntp = tpfpfntp.T
tpfpfntp.columns = ['TN','FN','TP','FP']
plt.figure()
sns.heatmap(cm_test.astype(int), annot=True,fmt='d', cmap="YlGnBu",xticklabels=names,yticklabels=names)

acc_test = accuracy_score(y_test,y_pred_test)

cce = keras.losses.SparseCategoricalCrossentropy()
loss = cce(enc_test, enc_pred).numpy()
#%%  ROC curves
fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()
for i in range(0,5):
    fpr_test[i], tpr_test[i], _ = roc_curve(enc_test[:, i], y_hat_test[:, i])
    roc_auc_test[i] = auc(fpr_test[i], tpr_test[i])
    

legend = ['berry AUC=%0.4f' %roc_auc_test[0],
          'bird AUC=%0.4f' %roc_auc_test[1],
          'dog AUC=%0.4f' %roc_auc_test[2],
          'flower AUC=%0.4f' %roc_auc_test[3],
          'other AUC=%0.4f' %roc_auc_test[4]]

plt.figure()
for i in range(0,5):
    plt.plot(fpr_test[i],tpr_test[i], lw=2, label=legend[i])
plt.legend()
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')

for i in range(0,5):
    plt.figure()
    plt.plot(fpr_test[i],tpr_test[i], lw=2, label=legend[i],color='r')
    plt.legend()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

#%% Saving the history of training
import pickle
with open('hist_loss', 'wb') as f:
        pickle.dump(hist_loss, f)
with open('hist_acc', 'wb') as f:
        pickle.dump(hist_acc, f)
        