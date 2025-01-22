# from sklearn.datasets import make_classification
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.svm import SVC
#
# x, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, n_classes=2, random_state=0)
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#
# tuned_parameters = [
#     {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
#     {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
# ]
#
# grid_search = GridSearchCV(
#     SVC(), tuned_parameters
# )
# grid_search.fit(x_train, y_train)
