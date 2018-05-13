import os
import sys
import shutil
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('common')
sys.path.append('external')
import util

from clr_callback import CyclicLR

import keras
import keras.backend as K
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Activation, Dropout, BatchNormalization, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard

class Dataset:
    def __init__(self, datadir, X_modifier=None):
        self.datadir = datadir
        self.load(X_modifier)
    def load(self, X_modifier=None):
        X_train_org = np.load(os.path.join(self.datadir, 'X_train.npy'))
        X_test = np.load(os.path.join(self.datadir, 'X_test.npy'))
        y_labels_train = pd.read_csv(os.path.join(self.datadir, 'y_train.csv'), sep=',')['scene_label'].tolist()
        # Make label lists
        self.labels = sorted(list(set(y_labels_train)))
        self.label2int = {l:i for i, l in enumerate(self.labels)}
        self.int2label = {i:l for i, l in enumerate(self.labels)}
        self.num_classes = len(self.labels)
        # Map y_train to int labels
        y_train_org = keras.utils.to_categorical([self.label2int[l] for l in y_labels_train])

        # Train/Validation split to X_train/y_train
        splitlist = pd.read_csv(os.path.join(self.datadir, 'crossvalidation_train.csv'), sep=',')['set'].tolist()
        X_train = np.array([x for i, x in enumerate(X_train_org) if splitlist[i] == 'train'])
        X_valid = np.array([x for i, x in enumerate(X_train_org) if splitlist[i] == 'test'])
        y_train = np.array([y for i, y in enumerate(y_train_org) if splitlist[i] == 'train'])
        y_valid = np.array([y for i, y in enumerate(y_train_org) if splitlist[i] == 'test'])
        
        # Special X handling
        if X_modifier:
            X_train, X_valid, X_test = X_modifier(X_train, X_valid, X_test)
            print('Applied special X modifier')

        # Normalize dataset
        value_max = np.max(np.vstack([X_train, X_valid, X_test]))
        X_train = X_train / value_max
        X_valid = X_valid / value_max
        X_test = X_test / value_max

        # [:, 40, 501] -> [:, 40, 501, 1]
        self.X_train = X_train[..., np.newaxis]
        self.X_valid = X_valid[..., np.newaxis]
        self.X_test = X_test[..., np.newaxis]
        self.y_train = y_train
        self.y_valid = y_valid

class Trainer:
    def __init__(self, name, d, model, lr, epochs, batch_size,
                use_cyclic_lr=True, use_random_eraser=True, use_mixup=True):
        self.name = name
        self.d = d
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_random_eraser = use_random_eraser
        self.use_mixup = use_mixup
        self.callbacks = [
            ModelCheckpoint('%s/best.h5' % name,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True),
            keras.callbacks.TensorBoard(log_dir='%s/log%s' % (name, name),
                                        histogram_freq=0, write_graph=True, write_images=True)
        ]
        if use_cyclic_lr:
            self.callbacks.append(
                CyclicLR(base_lr=lr, max_lr=lr * 10,
                         step_size=d.X_train.shape[0] // batch_size, mode='triangular'))
            print('using cyclic lr')
        else:
            print('not using cyclic lr')
        self.get_datagen()
        util.ensure_folder(name)
    def get_datagen(self):
        self.train_flow, self.valid_flow, self.test_flow, self.datagen, self.testdatagen = \
            get_datagen(self.d, self.batch_size,
                        use_random_eraser=self.use_random_eraser,
                        use_mixup=self.use_mixup)
    def fit(self):
        if os.path.exists('%s/log%s' % (self.name, self.name)):
            shutil.rmtree('%s/log%s' % (self.name, self.name))
        self.model.fit_generator(self.train_flow,
                            steps_per_epoch=self.d.X_train.shape[0] // self.batch_size,
                            epochs=self.epochs,
                            validation_data=self.valid_flow, callbacks=self.callbacks)
    def fine_tune(self, epochs=10):
        self.model.load_weights('%s/best.h5' % self.name)
        K.set_value(self.model.optimizer.lr, self.lr / 10)
        self.model.fit_generator(self.train_flow,
                            steps_per_epoch=self.d.X_train.shape[0] // self.batch_size,
                            epochs=epochs,
                            validation_data=self.valid_flow, callbacks=None)
        self.model.save_weights('%s/best.h5' % self.name)
    def get_valid_result(self, subname=''):
        self.get_datagen()
        self.model.load_weights('%s/best.h5' % self.name)
        valid_preds = self.model.predict_generator(self.valid_flow)
        np.save('%s/valid_preds%s.npy' % (self.name, subname), valid_preds)
        valid_pred_cls = [np.argmax(pred) for pred in valid_preds]
        valid_refs = [np.argmax(y) for y in self.d.y_valid]
        valid_results = [result == ref for result, ref in zip(valid_pred_cls, valid_refs)]
        acc = np.sum(valid_results)/len(valid_results)
        print(self.name, 'valid acc =', acc)
        return acc
    def write_test_result(self, subname=''):
        self.get_datagen()
        self.model.load_weights('%s/best.h5' % self.name)
        preds = self.model.predict_generator(self.test_flow)
        np.save('%s/test_preds%s.npy' % (self.name, subname), preds)
        filename = '%s/submit_%s%s.csv' % (self.name, self.name, subname)
        with open(filename, 'w') as f:
            f.writelines(['Id,Scene_label\n'])
            f.writelines(['%d,%s\n' % (i, self.d.int2label[np.argmax(pred)]) for i, pred in enumerate(preds)])
        print('wrote to', filename)

def after_fit(trainer, model):
    model.summary()
    trainer.get_valid_result()
    trainer.write_test_result()
def fine_tune(trainer, model):
    trainer.fine_tune(epochs=10)
    trainer.get_valid_result('_finetuned')
    trainer.write_test_result('_finetuned')

from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser

def get_datagen(dataset, batch_size, use_random_eraser=True, use_mixup=True):
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.6,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        preprocessing_function=\
            get_random_eraser(v_l=np.min(dataset.X_train), v_h=np.max(dataset.X_train)) \
            if use_random_eraser else None
    )
    test_datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
    )
    datagen.fit(np.r_[dataset.X_train, dataset.X_valid, dataset.X_test])
    test_datagen.mean, test_datagen.std = datagen.mean, datagen.std

    train_flow = datagen.flow(dataset.X_train, dataset.y_train, batch_size=batch_size)
    if use_mixup:
        train_flow = MixupGenerator(dataset.X_train, dataset.y_train, alpha=1.0, 
                                    batch_size=batch_size, datagen=datagen)()
    valid_flow = test_datagen.flow(dataset.X_valid, dataset.y_valid, shuffle=False)
    y_test_just_for_api = keras.utils.to_categorical(np.ones(len(dataset.X_test)))
    test_flow = test_datagen.flow(dataset.X_test, y_test_just_for_api, shuffle=False)
    return train_flow, valid_flow, test_flow, datagen, test_datagen

def plot_data(X, title_pattern, offset=0, N=20):
    fig, ax = plt.subplots(N//5, 5, figsize = (16, 2*N//5))
    for i in range(N):
        ax[i//5, i%5].pcolormesh(X[i + offset][..., -1])
        ax[i//5, i%5].set_title(title_pattern % (i + offset))
        ax[i//5, i%5].get_xaxis().set_ticks([])
    plt.show()