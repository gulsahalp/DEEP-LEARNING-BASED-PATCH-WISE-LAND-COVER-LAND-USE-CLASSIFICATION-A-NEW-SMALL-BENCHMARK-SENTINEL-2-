import tensorflow 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input, Add, Dense, Activation, Reshape
from tensorflow.keras.layers import Conv2D,Conv3D,ZeroPadding2D, MaxPooling2D, Flatten,MaxPooling3D, AveragePooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.applications import ResNet50,InceptionResNetV2
 

#--------------------------------------------------------------------------------------------

model = ResNet50(include_top="False", weights=None, input_shape=(100,100,3), pooling="max", classifier_activation="softmax")

#---------------------------------------------------------------------------------------------

# Store the fully connected layers
conv5_block3_out = model.layers[-3]
avg_pool = model.layers[-2]
predictions = model.layers[-1]

# Create the dropout layers
dropout = Dropout(0.50)

# Reconnect the layers
x = dropout(avg_pool.output)
predictors = Dense(7, activation='softmax', name='predictions')(x)



# Create a new model
model1 = Model(inputs=model.input, outputs=predictors)
model1.summary()

#---------------------------------------------Compile and Train The Model-------------------------------

model1.compile(loss="categorical_crossentropy", optimizer="nadam",
               metrics=["Accuracy"])

model1.fit(x_train, y_train, batch_size=16 , epochs=100, verbose = 1, validation_data=(x_valid, y_valid))


