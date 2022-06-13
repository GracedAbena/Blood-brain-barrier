
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from sklearn import svm 
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers, initializers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import plot_confusion_matrix as pcm
from chainer_chemistry.dataset.splitters import RandomSplitter
from chainer_chemistry.dataset.splitters import ScaffoldSplitter
from chainer_chemistry.dataset.splitters import StratifiedSplitter
from chainer_chemistry.training.extensions.roc_auc_evaluator import ROCAUCEvaluator
from chainer.iterators import SerialIterator
import sklearn.metrics as metric
from sklearn.preprocessing import StandardScaler


#---------Setting environments for reproducable results--------
seed_value = 101
os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

from tensorflow.compat.v1.keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1) 
                              
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)
#---------End of setting environments-----------------------

def normalize_data(d, _range, min_max=[-1000,1000]):
    if min_max[0] != -1000:
        n_d = ((_range[1]-_range[0])*((d-min_max[0]) / (min_max[1]-min_max[0]))) + np.full(d.shape, _range[0], dtype='float64')
    else:            
        n_d = ((_range[1]-_range[0])*((d-np.min(d)) / (np.max(d)-np.min(d)))) + np.full(d.shape, _range[0], dtype='float64')
    return n_d

def select_data(dx, dy, inds):
    new_d = np.zeros(dx.shape)
    new_y = np.zeros(dy.shape)
    
    for i in range(inds.shape[0]):
        _id = inds[i]
        new_d[_id,] = dx[_id,]
        new_y[_id,] = dy[_id,]
        
    new_d = new_d[:inds.shape[0], ]
    new_y = new_y[:inds.shape[0], ]
    
    return new_d, new_y

sol = pd.read_csv('BBBP_example.csv')#BBBP.csv   2nd_data.csv

def generate(smiles, lab, verbose=False):
    smiles_list = smiles.tolist()
    smiles_select = []
    moldata= []
    label = []
    lab_list = lab.tolist()
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
        
    baseData= np.arange(1,1)
    i=0  
    for mol in moldata:        
       
        try:
            desc_MolLogP = Descriptors.MolLogP(mol)
            desc_MolWt = Descriptors.MolWt(mol)
            desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
               
            row = np.array([desc_MolLogP,
                            desc_MolWt,
                            desc_NumRotatableBonds])   
        
            if(i==0):
                baseData=row
            else:
                baseData=np.vstack([baseData, row])
            i=i+1                           
        except:            
            continue
        ind = moldata.index(mol)
        label.append(lab_list[ind])
        smiles_select.append(smiles_list[ind]) 
        
    columnNames=["MolLogP","MolWt","NumRotatableBonds"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    descriptors = descriptors.values
    label = np.array(label)
    
    return descriptors, label, smiles_select

#X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=45)
#score = accuracy_score(Y_test ,pred_Y)

############################################################

def molToImg(smiles, lab, img_shape):
    smiles_list = smiles.tolist()
    lab_list = lab.tolist()
    imgs_list = []
    label = []
    for s in range(len(smiles_list)):        
        try:
            mol = Chem.MolFromSmiles(smiles_list[s])
            img = Draw.MolToImage(mol, img_shape)
            img = np.asarray(img, dtype='float64')
            imgs_list.append(img)
            
            ind = lab_list[s]
            label.append(ind)
        except:
            print(s)
            continue
    imgs_list = np.array(imgs_list)
    label = np.array(label)

    return imgs_list, label

############## Descriptors processing #############
    
X, Y, smiles_list = generate(sol.smiles, sol.p_np)

#scaler = StandardScaler()
#scaler.fit(X.tolist())
#X = scaler.transform(X.tolist())
X[:,0:1] = normalize_data(X[:,0:1], _range=[0,1])
X[:,1:2] = normalize_data(X[:,1:2], _range=[0,1])
X[:,2:3] = normalize_data(X[:,2:3], _range=[0,1])

splitter = RandomSplitter()

train_id, valid_id, test_id = splitter.train_valid_test_split(X, frac_train=0.7, frac_valid=0.3, frac_test=0.0, return_index=True, seed=45)
                                                #frac_test=0.10, test_id  

x_train, y_train = select_data(dx=X, dy=Y, inds=train_id)
x_valid, y_valid = select_data(dx=X, dy=Y, inds=valid_id)
x_test, y_test = select_data(dx=X, dy=Y, inds=test_id)

############ Image processing ##############

X_img, Y_img = molToImg(sol.smiles, sol.p_np, (1000,1000))
X_img = X_img / np.max(X_img)

l_encoder = OneHotEncoder() 
l_encoder.fit(np.reshape(Y_img, (Y_img.size,1)))     
Y_img = l_encoder.transform(np.reshape(Y_img, (Y_img.size,1)))
Y_img = Y_img.toarray()

x_train_img, y_train_img = select_data(dx=X_img, dy=Y_img, inds=train_id)
x_valid_img, y_valid_img = select_data(dx=X_img, dy=Y_img, inds=valid_id)
x_test_img, y_test_img = select_data(dx=X_img, dy=Y_img, inds=test_id)

def CNN(inpt_shape, fusion=False):
    
    inpt = layers.Input(shape=inpt_shape)
    
    m = layers.Conv2D(filters=16, kernel_size=(4,4), activation='relu', padding="same")(inpt)
    #m = layers.Conv2D(filters=16, kernel_size=(4,4), activation='relu', padding="same")(m)
    m = layers.MaxPool2D(pool_size=(2,2), padding="same")(m)
    
    m = layers.Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding="same")(m)
    #m = layers.Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding="same")(m)
    m = layers.MaxPool2D(pool_size=(2,2), padding="same")(m)
    
    m = layers.Conv2D(filters=64, kernel_size=(4,4), activation='relu', padding="same")(m)
    #m = layers.Conv2D(filters=64, kernel_size=(4,4), activation='relu', padding="same")(m)
    m = layers.MaxPool2D(pool_size=(2,2), padding="same")(m)
    
    m = layers.Conv2D(filters=128, kernel_size=(4,4), activation='relu', padding="same")(m)
    #m = layers.Conv2D(filters=128, kernel_size=(4,4), activation='relu', padding="same")(m)
    m = layers.MaxPool2D(pool_size=(2,2), padding="same")(m)
    
    m = layers.Conv2D(filters=256, kernel_size=(4,4), activation='relu', padding="same")(m)
    #m = layers.Conv2D(filters=256, kernel_size=(4,4), activation='relu', padding="same")(m)
    m = layers.MaxPool2D(pool_size=(2,2), padding="same")(m)
    
    m = layers.Conv2D(filters=512, kernel_size=(4,4), activation='relu', padding="same")(m)
    #m = layers.Conv2D(filters=512, kernel_size=(4,4), activation='relu', padding="same")(m)
    m = layers.MaxPool2D(pool_size=(2,2), padding="same")(m)
    
    m = layers.GlobalAveragePooling2D()(m)
    
    m = layers.Dense(64, activation='relu')(m)
    m = layers.Dense(32, activation='relu')(m)
    m = layers.Dense(8, activation='relu')(m)
    
    if fusion==False:
        m = layers.Dense(2, activation="softmax")(m)
    
    m = Model(inpt, m)
    return m

cnn_model = CNN(x_train_img[0].shape)

cnn_model.summary()

_adam = optimizers.Adam(lr=0.00002, beta_1=0.5)
cnn_model.compile(loss="categorical_crossentropy", optimizer=_adam, metrics=["accuracy"])
cnn_checkpoint = ModelCheckpoint("cnn_model.h5", monitor='val_accuracy',verbose=1,save_best_only=True, mode='max', save_freq='epoch')

cnn_hist = cnn_model.fit(x_train_img, y_train_img, batch_size=16, epochs=50, validation_data=(x_valid_img, y_valid_img),
                         callbacks=[cnn_checkpoint])

cnn_model = load_model("cnn_model.h5")

cnn_pred = cnn_model.predict(x_train_img, batch_size=16)
cnn_pred = np.argmax(cnn_pred, axis=1)
cnn_true = np.argmax(y_train_img, axis=1)
print("CNN Train Accuracy :", metric.accuracy_score(cnn_true ,cnn_pred))
print("CNN Train ROC_AUC :", metric.roc_auc_score(cnn_true ,cnn_pred))
fpr, tpr, thresholds = metric.roc_curve(cnn_true, cnn_pred)
print("CNN Train AUC :", metric.auc(fpr, tpr))
print("CNN Train F1-Score :", metric.f1_score(cnn_true ,cnn_pred))

cnn_pred = cnn_model.predict(x_test_img, batch_size=16)
cnn_pred = np.argmax(cnn_pred, axis=1)
cnn_true = np.argmax(y_test_img, axis=1)
print("CNN Test Accuracy :", metric.accuracy_score(cnn_true ,cnn_pred))
print("CNN Test ROC_AUC :", metric.roc_auc_score(cnn_true ,cnn_pred))
fpr, tpr, thresholds = metric.roc_curve(cnn_true, cnn_pred)
print("CNN Test AUC :", metric.auc(fpr, tpr))
print("CNN Test F1-Score :", metric.f1_score(cnn_true ,cnn_pred))

cnn_pred = cnn_model.predict(x_valid_img, batch_size=16)
cnn_pred = np.argmax(cnn_pred, axis=1)
cnn_true = np.argmax(y_valid_img, axis=1)
print("CNN Valid Accuracy :", metric.accuracy_score(cnn_true ,cnn_pred))
print("CNN Valid ROC_AUC :", metric.roc_auc_score(cnn_true ,cnn_pred))
fpr, tpr, thresholds = metric.roc_curve(cnn_true, cnn_pred)
print("CNN Valid AUC :", metric.auc(fpr, tpr))
print("CNN Valid F1-Score :", metric.f1_score(cnn_true ,cnn_pred))

def FC(fusion=False):
    
    inpt = layers.Input(shape=(3))
    
    m = layers.Dense(256, activation='tanh')(inpt)
    m = layers.Dense(128, activation='tanh')(m)
    m = layers.Dense(64, activation='tanh')(m)
    m = layers.Dense(32, activation='tanh')(m)
    m = layers.Dense(16, activation='tanh')(m)
    m = layers.Dense(8, activation='tanh')(m)
    
    if fusion==False:
        m = layers.Dense(2, activation="softmax")(m)
    
    m = Model(inpt, m)
    return m
    
fc_model = FC()

fc_model.summary()

_adam = optimizers.Adam(lr=0.00002, beta_1=0.5)
fc_model.compile(loss="categorical_crossentropy", optimizer=_adam, metrics=["accuracy"])
fc_checkpoint = ModelCheckpoint("fc_model.h5", monitor='val_accuracy',verbose=1,save_best_only=True, mode='max', save_freq='epoch')

fc_hist = fc_model.fit(x_train, y_train_img, batch_size=16, epochs=50, validation_data=(x_valid, y_valid_img), callbacks=[fc_checkpoint])   

fc_model = load_model("fc_model.h5")

fc_pred = fc_model.predict(x_train, batch_size=16)
fc_pred = np.argmax(fc_pred, axis=1)
fc_true = np.argmax(y_train_img, axis=1)
print("FC Train Accuracy :", metric.accuracy_score(fc_true ,fc_pred))
print("FC Train ROC_AUC :", metric.roc_auc_score(fc_true ,fc_pred))
fpr, tpr, thresholds = metric.roc_curve(fc_true ,fc_pred)
print("FC Train AUC :", metric.auc(fpr, tpr))
print("FC Train F1-Score :", metric.f1_score(fc_true ,fc_pred))

fc_pred = fc_model.predict(x_test, batch_size=16)
fc_pred = np.argmax(fc_pred, axis=1)
fc_true = np.argmax(y_test_img, axis=1)
print("FC Test Accuracy :", metric.accuracy_score(fc_true ,fc_pred))
print("FC Test ROC_AUC :", metric.roc_auc_score(fc_true ,fc_pred))
fpr, tpr, thresholds = metric.roc_curve(fc_true ,fc_pred)
print("FC Test AUC :", metric.auc(fpr, tpr))
print("FC Test F1-Score :", metric.f1_score(fc_true ,fc_pred))

fc_pred = fc_model.predict(x_valid, batch_size=16)
fc_pred = np.argmax(fc_pred, axis=1)
fc_true = np.argmax(y_valid_img, axis=1)
print("FC Valid Accuracy :", metric.accuracy_score(fc_true ,fc_pred))
print("FC Valid ROC_AUC :", metric.roc_auc_score(fc_true ,fc_pred))
fpr, tpr, thresholds = metric.roc_curve(fc_true ,fc_pred)
print("FC Valid AUC :", metric.auc(fpr, tpr))
print("FC Valid F1-Score :", metric.f1_score(fc_true ,fc_pred))


############# Fusion ###################

mod_1 = CNN(x_train_img[0].shape, fusion=True)
mod_2 = FC(fusion=True)

fuse_model = layers.Concatenate()([mod_1.output, mod_2.output])
fuse_model = layers.Dense(2, activation="softmax")(fuse_model)
fuse_model = Model([mod_1.input, mod_2.input] ,fuse_model)

fuse_model.summary()

_adam = optimizers.Adam(lr=0.00002, beta_1=0.5)
fuse_model.compile(loss="categorical_crossentropy", optimizer=_adam, metrics=["accuracy"])
class_weight = {0:100, 1:1}
fuse_checkpoint = ModelCheckpoint("fuse_model.h5", monitor='val_accuracy',verbose=1,save_best_only=True, mode='max', save_freq='epoch')

fuse_hist = fuse_model.fit([x_train_img, x_train], y_train_img, batch_size=16, epochs=20, 
                           validation_data=([x_valid_img, x_valid], y_valid_img),
                           callbacks=[fuse_checkpoint])# class_weight=class_weight, 

fuse_model = load_model("fuse_model.h5")
fuse_pred = fuse_model.predict([x_test_img, x_test])
fuse_pred = np.argmax(fuse_pred, axis=1)
fuse_true = np.argmax(y_test_img, axis=1)
print("FC Test accuracy :", roc_auc_score(fuse_true, fuse_pred))

fuse_pred = fuse_model.predict([x_valid_img, x_valid])
fuse_pred = np.argmax(fuse_pred, axis=1)
fuse_true = np.argmax(y_valid_img, axis=1)
print("FC Validation accuracy :", roc_auc_score(fuse_true, fuse_pred))

#print(confusion_matrix(y_true=fuse_true ,y_pred=fuse_pred, normalize='true'))

class_names = ["inactive","active"]
class_names = np.array(class_names).T

pcm.plot_confusion_matrix(fuse_true, fuse_pred, classes=class_names, normalize=False,
                      title='', _dpi=100, font_size=7)




###############################################################
#################################################################
##################################################################

splts = 1000
from sklearn.model_selection import ShuffleSplit
shuffle = ShuffleSplit(n_splits=splts, test_size=0.3, random_state=42)

############## Descriptors processing #############
    
X, Y, smiles_list = generate(sol.smiles, sol.p_np)

X[:,0:1] = normalize_data(X[:,0:1], _range=[0,1])
X[:,1:2] = normalize_data(X[:,1:2], _range=[0,1])
X[:,2:3] = normalize_data(X[:,2:3], _range=[0,1])

l_encoder = OneHotEncoder() 
l_encoder.fit(np.reshape(Y, (Y.size,1)))     
Y = l_encoder.transform(np.reshape(Y, (Y.size,1)))
Y = Y.toarray()
############ Image processing ##############

X_img, Y_img = molToImg(sol.smiles, sol.p_np, (200,200))
X_img = X_img / np.max(X_img)

l_encoder = OneHotEncoder() 
l_encoder.fit(np.reshape(Y_img, (Y_img.size,1)))     
Y_img = l_encoder.transform(np.reshape(Y_img, (Y_img.size,1)))
Y_img = Y_img.toarray()

###### Train FC for 1000 splits #########

results = np.zeros((splts,9))
ind = 0
for train_id, test_id in shuffle.split(X_img):
    x_train, y_train = select_data(dx=X_img, dy=Y_img, inds=train_id)
    x_test, y_test = select_data(dx=X_img, dy=Y_img, inds=test_id)
    '''
    fc_model = FC()    
    fc_model.summary()    
    _adam = optimizers.Adam(lr=0.00002, beta_1=0.5)
    fc_model.compile(loss="categorical_crossentropy", optimizer=_adam, metrics=["accuracy"])
    fc_checkpoint = ModelCheckpoint("fc_model.h5", monitor='val_accuracy',verbose=1,save_best_only=True, mode='max', save_freq='epoch')
    
    fc_hist = fc_model.fit(x_train, y_train, batch_size=16, epochs=50, validation_data=(x_test, y_test), callbacks=[fc_checkpoint])   
    fc_model = load_model("fc_model.h5")
    '''
    
    cnn_model = CNN(x_train[0].shape)

    cnn_model.summary()
    
    _adam = optimizers.Adam(lr=0.00002, beta_1=0.5)
    cnn_model.compile(loss="categorical_crossentropy", optimizer=_adam, metrics=["accuracy"])
    cnn_checkpoint = ModelCheckpoint("cnn_model.h5", monitor='val_accuracy',verbose=1,save_best_only=True, mode='max', save_freq='epoch')
    
    cnn_hist = cnn_model.fit(x_train, y_train, batch_size=16, epochs=50, validation_data=(x_test, y_test),
                             callbacks=[cnn_checkpoint])
    
    cnn_model = load_model("cnn_model.h5")

    fc_pred = cnn_model.predict(x_train, batch_size=16)
    fc_pred = np.argmax(fc_pred, axis=1)
    fc_true = np.argmax(y_train, axis=1)   
    
    results[ind,0] = metric.accuracy_score(fc_true ,fc_pred)
    results[ind,1] = metric.roc_auc_score(fc_true ,fc_pred)
    fpr, tpr, thresholds = metric.roc_curve(fc_true ,fc_pred)
    results[ind,2] = metric.auc(fpr, tpr)
    results[ind,3] = metric.f1_score(fc_true ,fc_pred)
    
    fc_pred = cnn_model.predict(x_test, batch_size=16)
    fc_pred = np.argmax(fc_pred, axis=1)
    fc_true = np.argmax(y_test, axis=1)
    
    results[ind,5] = metric.accuracy_score(fc_true ,fc_pred)
    results[ind,6] = metric.roc_auc_score(fc_true ,fc_pred)
    fpr, tpr, thresholds = metric.roc_curve(fc_true ,fc_pred)
    results[ind,7] = metric.auc(fpr, tpr)
    results[ind,8] = metric.f1_score(fc_true ,fc_pred)
    
    ind = ind + 1
    
results = pd.DataFrame(results)
results.to_csv("results_cnn.csv")















