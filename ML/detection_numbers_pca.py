from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, precision_score, recall_score

digits = load_digits()

import matplotlib.pyplot as plt

# x = digits.images[700]
#
# print(digits.target[700])
# plt.gray()
# plt.imshow(x)
# plt.show()

# Preprocessing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=32)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


# Performance metric
def calculate_metrics(y_tr, y_te, y_pr_tr, y_pr_te):
    acc_train = accuracy_score(y_true=y_tr, y_pred=y_pr_tr)

    acc_test = accuracy_score(y_true=y_te, y_pred=y_pr_te)
    prec = precision_score(y_true=y_te, y_pred=y_pr_te, average="weighted")
    re = recall_score(y_true=y_te, y_pred=y_pr_te, average="weighted")

    print(f"acc train: {acc_train} - acc test: {acc_test} - precision: {prec} - recall: {re}")
    return acc_train, acc_test, prec, re


# Classification
# 1. Random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=128, n_estimators=256)
rf.fit(x_train, y_train)

y_pred_train = rf.predict(x_train)
y_pred_test = rf.predict(x_test)

acc_train_rf, acc_test_rf, p_rf, r_rf = calculate_metrics(y_train, y_test, y_pred_train, y_pred_test)

# 2. SVM
from sklearn.svm import SVC

svm = SVC(kernel="poly")
svm.fit(x_train, y_train)

y_pred_train = svm.predict(x_train)
y_pred_test = svm.predict(x_test)

acc_train_svm, acc_test_svm, p_svm, r_svm = calculate_metrics(y_train, y_test, y_pred_train, y_pred_test)

# 3. ANN
from sklearn.neural_network import MLPClassifier

ann = MLPClassifier(hidden_layer_sizes=256, batch_size="auto", learning_rate="adaptive")
ann.fit(x_train, y_train)

y_pred_train = ann.predict(x_train)
y_pred_test = ann.predict(x_test)

acc_train_ann, acc_test_ann, p_ann, r_ann = calculate_metrics(y_train, y_test, y_pred_train, y_pred_test)

# 4. KNN(last)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train, y_train)

y_pred_train = knn.predict(x_train)
y_pred_test = knn.predict(x_test)

acc_train_knn, acc_test_knn, p_knn, r_knn = calculate_metrics(y_train, y_test, y_pred_train, y_pred_test)

# Comparison
accuracy_train = [acc_train_knn, acc_train_rf, acc_train_svm, acc_train_ann]
title = ["KNN", "RF", "SVM", "ANN"]
colors = ["black", "red", "yellow", "orange"]

plt.bar(title, accuracy_train, color=colors)
plt.grid()
plt.show()

accuracy_test = [acc_test_knn, acc_test_rf, acc_test_svm, acc_test_ann]
plt.bar(title, accuracy_test, color=colors)
plt.grid()
plt.show()

p = [p_knn, p_rf, p_svm, p_ann]
plt.bar(title, p, color=colors)
plt.grid()
plt.show()

r = [r_knn, r_rf, r_svm, r_ann]
plt.bar(title, r, color=colors)
plt.grid()
plt.show()
