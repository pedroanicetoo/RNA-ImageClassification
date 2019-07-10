import numpy as np 
from sklearn.metrics import confusion_matrix
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.layers import  MaxPooling2D, Dropout
  
import itertools
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')
from sklearn.metrics import classification_report

train_path= ('./dice/train') ##dice-d4-d6-d8-d10-d12-d20
valid_path= ('./dice/valid') ##dice-d4-d6-d8-d10-d12-d20

batch_size_train=10
batch_size_valid=10
targetsize= 48

#Obtendo os conjuntos de treino e teste
train_batches= ImageDataGenerator().flow_from_directory(train_path, target_size=(targetsize,targetsize), classes=['d4', 'd6', 'd8', 'd10','d12','d20'],batch_size= batch_size_train)
valid_batches= ImageDataGenerator().flow_from_directory(valid_path, target_size=(targetsize,targetsize), classes=['d4', 'd6', 'd8', 'd10','d12','d20'],batch_size= batch_size_valid)

train_num = len(train_batches)
val_num = len(valid_batches) 

def plots(ims, figsize=(20,10), rows=1, interp= False, titles= None):
	if type(ims[0]) is np.ndarray:
		ims = np.array(ims).astype(np.uint8)
		if (ims.shape[-1] != 3):
			ims= ims.transpose((0,1,2,3))
	f= plt.figure(figsize=figsize)
	cols= len(ims)//rows if len(ims) %2 == 0 else len(ims)//rows + 1
	for i in range(len(ims)):
		sp = f.add_subplot(rows, cols, i+1)
		sp.axis('Off')
		if titles is not None:
			sp.set_title(titles[i], fontsize=12)
		plt.imshow(ims[i], interpolation=None if interp else 'none')

##Imagens ilustrando as classes d4, d6, d8, d10, d12 e d20;
imgs, labels = next(train_batches)
plots(imgs, titles = labels)


#primeira camada(Flatten) transforma a matriz 48x48
#primeira camada(Dense) tem 512 neuronios
#A segunda camada softmax com 6 nodos - isto retorna um vetor com 6 valores de probabilidade que soma 1
#Cada no contem o valor que indica a probabilidade que a image atual pertence a uma das 6 classes.
model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(targetsize,targetsize, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
  
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
  
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
  
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax'),
  
    ])

model.summary()

#camadas do modelo
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics= ['accuracy'])
history = model.fit_generator(train_batches, steps_per_epoch= train_num ,
					validation_data=valid_batches, validation_steps= val_num, epochs=15, verbose=2)
#Curva de Acurácia
plt.figure()
#plt.plot(np.arange(0,epochs), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0,epochs), H.history["val_loss"], label="val_loss")
plt.plot(history.history["acc"],"r",linewidth=3.0)
plt.plot(history.history["val_acc"],"b",linewidth=3.0)
plt.title("Treinamento de Acurácia / Validação de Acurácia")
plt.xlabel("Épocas #")
plt.ylabel("Acurácia")
plt.legend()
plt.show()

# Curvas de perdas
plt.figure()
#plt.plot(np.arange(0,epochs), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0,epochs), H.history["val_loss"], label="val_loss")
plt.plot(history.history["loss"],"r",linewidth=3.0)
plt.plot(history.history["val_loss"],"b",linewidth=3.0)
plt.title("Treinamento de perdas / Validação de perdas")
plt.xlabel("Épocas #")
plt.ylabel("Perdas")
plt.legend()
plt.show()

