# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 22:30:52 2020

@author: MSc. Maciej Rosoł
"""

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense,Flatten,Dropout,GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import tensorflow as tf
import keras
from sklearn.model_selection import KFold
from tensorflow.keras import regularizers
import timeit
from tensorflow.keras import backend as K
import os
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

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
    
#%% Initializing list and dictionaries
X = np.asarray(images)
y = np.asanyarray(labels).reshape(-1, 1)
kf = KFold(5,shuffle=True,random_state=10)    
hist = {}
time = {}
y_hat_train = []
y_hat_test = []
acc_train = []
acc_test = []
cm_train = []
cm_test = []
#%% Functions definitions
def init_base_model(): #function for loading the base model of ResNet50
    base_model = ResNet50(include_top=False, input_shape=(256, 256, 3))
    for j,layer in enumerate(base_model.layers):
        if j<len(base_model.layers)-4:
            layer.trainable=False
    return base_model

def create_model(): # creating the final model based on ResNet50
    base_model = init_base_model()
    mdl = base_model.output
    mdl = GlobalAveragePooling2D()(mdl)
    mdl = Dense(1024,activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2))(mdl) 
    mdl = Dropout(0.05)(mdl)
    mdl = Dense(1024,activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-2, l2=1e-2))(mdl)
    mdl = Dropout(0.05)(mdl)
    mdl = Dense(5,activation='softmax')(mdl)
    mdl = Model(inputs=base_model.inputs, outputs=mdl)
    return mdl

def train(mdl,lr,X_train,y_train): # Training the model with the given learning rate, all other hyperparameters are set as default
    opt = keras.optimizers.Adam(learning_rate=lr)
    mdl.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = mdl.fit(X_train,y_train,
                 epochs=5,
                 batch_size=32,
                 shuffle=True,
                 validation_data=(X_test,y_test),
                 verbose=True)
    return history
#%% K-Fold cross validation
with tf.device('/gpu:0'):
    i = 0
    for train_idx, test_idx in kf.split(X,y): # summary 20 epochs after each 5 epochs the learning rate is decreasing
        time[i] = []
        hist[i] = []
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = create_model()
    
        time[i].append(timeit.default_timer())
        
        history1 = train(model,0.001,X_train,y_train) 

        hist[i].append(history1)
        
        time[i].append(timeit.default_timer())
        
        history2 = train(model,0.0001,X_train,y_train)
        
        hist[i].append(history2)
        
        time[i].append(timeit.default_timer())
        
        history3 = train(model,0.00001,X_train,y_train)
        
        hist[i].append(history3)
        
        time[i].append(timeit.default_timer())
        
        history4 = train(model,0.000001,X_train,y_train)
        
        hist[i].append(history4)
        
        time[i].append(timeit.default_timer())
        
        h_acc = []
        h_acc_val = []
        h_loss = []
        h_loss_val = []
        for h in hist[i]: 
            h_acc.append(h.history['accuracy'])
            h_acc_val.append(h.history['val_accuracy'])
            h_loss.append(h.history['loss'])
            h_loss_val.append(h.history['val_loss'])
        # plotting traing accuracy and loss
        plt.figure()
        plt.plot(np.hstack(h_acc))
        plt.plot(np.hstack(h_acc_val))
        plt.legend(['train','val'])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.figure()
        plt.plot(np.hstack(h_loss))
        plt.plot(np.hstack(h_loss_val))
        plt.legend(['train','val'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        
        # prediction on the train and test sets, calculating the confusion matrices
        y_hat_train = model.predict(X_train)
        y_pred_train = np.argmax(y_hat_train,axis=1)
        acc_train.append(accuracy_score(y_train,y_pred_train))
        cm_train.append(confusion_matrix(y_train,y_pred_train))
        
        y_hat_test = model.predict(X_test)
        y_pred_test = np.argmax(y_hat_test,axis=1)
        acc_test.append(accuracy_score(y_test,y_pred_test))
        cm_test.append(confusion_matrix(y_test,y_pred_test))
    
        model.save('model'+str(i)) # saving the model
        
        del model
        K.clear_session()
        
        i += 1 # next iteration of k-fold cross validaton
    
#%% Performance after k-fold validation
cm_train_all = np.zeros([5,5])
for cm in cm_train:
    cm_train_all += cm # cumulative confusion matrix for train data
    
cm_test_all = np.zeros([5,5])
for cm in cm_test:
    cm_test_all += cm # cumulative confusion matrix for test data
    
# plotting confusion matrices
plt.figure(figsize=[10,7])
plt.subplot(211)
sns.heatmap(cm_train_all.astype(int), annot=True,fmt='d', cmap="YlGnBu")
plt.subplot(212)
sns.heatmap(cm_test_all.astype(int), annot=True,fmt='d', cmap="YlGnBu")

print('Train accuracy = %0.2f +- %0.2f' %(np.mean(acc_train),np.std(acc_train)))

print('Test accuracy = %0.2f +- %0.2f' %(np.mean(acc_test),np.std(acc_test)))

#%% Saving variables
import pickle
with open('bic2.pkl', 'wb') as f:
    pickle.dump([time,acc_train,acc_test], f)

for l in range(0,5):
    h_acc = []
    h_val_acc = []
    h_loss = []
    h_val_loss = []
    for i in range(4):
        h_acc.append(hist[l][i].history['accuracy'][:])
        h_val_acc.append(hist[l][i].history['val_accuracy'][:])
        h_loss.append(hist[l][i].history['loss'][:])
        h_val_loss.append(hist[l][i].history['val_loss'][:])
    h_acc = np.hstack(h_acc)
    h_val_acc = np.hstack(h_val_acc)
    h_loss = np.hstack(h_loss)
    h_val_loss = np.hstack(h_val_loss)

    with open('hist_acc_%d'%l, 'wb') as f:
        pickle.dump(h_acc, f)
    with open('hist_val_acc_%d'%l, 'wb') as f:
        pickle.dump(h_val_acc, f)
    with open('hist_loss_%d'%l, 'wb') as f:
        pickle.dump(h_loss, f)
    with open('hist_val_loss_%d'%l, 'wb') as f:
        pickle.dump(h_val_loss, f)
