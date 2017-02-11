import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier


# import data and divided it into training and test purposes
iris = datasets.load_iris()
x = preprocessing.scale(iris.data)
y = np_utils.to_categorical(iris.target)
x_tr, x_te, y_tr, y_te = train_test_split(x, y, train_size  = 0.7)
num_classes = y_te.shape[1]


# Define model for iris classification
def iris_model(activation="relu", optimizer="adam", out_dim=100):
    model = Sequential()
    model.add(Dense(out_dim, input_dim=4, activation=activation))
    model.add(Dense(out_dim, activation=activation))   
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Define options for parameters
activation = ["relu", "sigmoid"]
optimizer = ["adam", "adagrad"]
out_dim = [100, 200]
nb_epoch = [10, 25]
batch_size = [5, 10]


# Retrieve model and parameter into GridSearchCV
model = KerasClassifier(build_fn=iris_model, verbose=0)
param_grid = dict(activation=activation, 
                  optimizer=optimizer, 
                  out_dim=out_dim, 
                  nb_epoch=nb_epoch, 
                  batch_size=batch_size)
grid = GridSearchCV(estimator=model, param_grid=param_grid)


# Run grid search
grid_result = grid.fit(x_tr, y_tr)


# Get the best score and the optimized mode
print (grid_result.best_score_)
print (grid_result.best_params_)

# Evaluate the model with test data
grid_eval = grid.predict(x_te)
def y_binary(i):
    if   i == 0: return [1, 0, 0]
    elif i == 1: return [0, 1, 0]
    elif i == 2: return [0, 0, 1]
y_eval = np.array([y_binary(i) for i in grid_eval])
accuracy = (y_eval == y_te)
print (np.count_nonzero(accuracy == True) / (accuracy.shape[0] * accuracy.shape[1]))


# Now see the optimized model
model = iris_model(activation=grid_result.best_params_['activation'], 
                   optimizer=grid_result.best_params_['optimizer'], 
                   out_dim=grid_result.best_params_['out_dim'])
model.summary()


