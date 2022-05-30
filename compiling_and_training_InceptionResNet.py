from tensorflow.python.keras.layers.convolutional import ZeroPadding1D
from tensorflow.python.keras.engine.input_layer import InputLayer
import tensorflow 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Dense, Activation,Reshape,Sequential, ZeroPadding2D,Concatenate 
from tensorflow.keras.layers import Conv2D,Conv3D,ZeroPadding2D, MaxPooling2D, Flatten,MaxPooling3D, AveragePooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import ResNet50, InceptionResNetV2


#--------------------------------------------------------------------------------------------
InceptionResNet= InceptionResNetV2(include_top="False", weights=None, input_shape=(100,100,3), pooling="max", classifier_activation="softmax")
InceptionResNet.summary()
#---------------------------------------------------------------------------------------------

#----------------------------------------InceptionResNet--------------------------------------------------

avg_pool = InceptionResNet.layers[-2]
predictions = InceptionResNet.layers[-1]

# Create the dropout layers
dropout = Dropout(0.50)
# Reconnect the layers
x = dropout(avg_pool.output)
predictors = Dense(7, activation='softmax', name='predictions')(x)

# Create a new model
model1 = Model(InceptionResNet.inputs, outputs=predictors)
model1.summary()

#---------------------------------------------Compile and Train The Model InceptionResNet-------------------------------

model1.compile(loss="categorical_crossentropy", optimizer=tensorflow.keras.optimizers.SGD(learning_rate=0.00005, momentum=0.9),
               metrics=["CategoricalAccuracy"])

model1.fit(x_train, y_train, batch_size=16 , epochs=100, verbose = 1, validation_data=(x_valid, y_valid))

