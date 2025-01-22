# datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV

x, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, n_classes=2, random_state=0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# model definition
import keras

def create_model1(hidden_units=16, optimizer="adam"):
  model1 = keras.Sequential()
  model1.add(keras.layers.Dense(units=hidden_units, input_shape=(None, 20), activation="relu"))
  model1.add(keras.layers.Dense(units=1, activation="sigmoid"))

  model1.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=["accuracy"])

  return model1


# Grid search
from scikeras.wrappers import KerasClassifier

model = KerasClassifier(model=create_model1, loss="binary_crossentropy", epochs=10, batch_size=32, verbose=0)

params = {
    "model__hidden_units": [16, 32, 64],
    "model__optimizer": ["adam", "rmsprop"]
}

grid = GridSearchCV(estimator=model, param_grid=params, cv=3)
grid_result = grid.fit(x_train, y_train)

best_params = grid_result.best_params_
best_score = grid_result.best_score_

# Random search
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split

rs = RandomizedSearchCV(estimator=model, param_distributions=params, cv=3, n_iter=5)
rs_result = rs.fit(x_train, y_train)

rs_best_params = rs_result.best_params_

rs_best_score = rs_result.best_score_
