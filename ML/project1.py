from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, recall_score, precision_score

bc = load_breast_cancer()

# Preprocessing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(bc.data, bc.target, test_size=0.2)

# print(f"Feature=> train: {x_train.shape}, test {x_test.shape}")
# print(f"Feature=> train: {y_train.shape}, test {y_test.shape}")

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Classification

# Comparison between Classification Metrics
def calculate_metrics(y_tr, y_te, y_pr_tr, y_pr_te):
    acc_train = accuracy_score(y_true=y_tr, y_pred=y_pr_tr)

    acc_test = accuracy_score(y_true=y_te, y_pred=y_pr_te)
    prec = precision_score(y_true=y_te, y_pred=y_pr_te)
    rec = recall_score(y_true=y_te, y_pred=y_pr_te)

    print(f"acc train: {acc_train} - acc test: {acc_test} - precision: {prec} - recall: {rec}")
    return acc_train, acc_test, prec, rec


# 1. Naive bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred_tr = gnb.predict(x_train)
y_pred_te = gnb.predict(x_test)

# acc_train_nb, acc_test_nb, prec_nb, rec_nb = calculate_metrics(y_train, y_test, y_pred_tr, y_pred_te)

# 2. KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=8, algorithm="kd_tree", leaf_size=28)
knn.fit(x_train, y_train)

y_pred_tr = knn.predict(x_train)
y_pred_te = knn.predict(x_test)

# acc_train_knn, acc_test_knn, prec_knn, rec_knn = calculate_metrics(y_train, y_test, y_pred_tr, y_pred_te)

# 3. Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=128, min_samples_split=4, criterion="gini")
dt.fit(x_train, y_train)

y_pred_tr = dt.predict(x_train)
y_pred_te = dt.predict(x_test)

# acc_train_dt, acc_test_dt, prec_dt, rec_dt = calculate_metrics(y_train, y_test, y_pred_tr, y_pred_te)

# 4. Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1000, max_depth=16, min_samples_split=2)
rf.fit(x_train, y_train)

y_pred_tr = rf.predict(x_train)
y_pred_te = rf.predict(x_test)

# acc_train_rf, acc_test_rf, prec_rf, rec_rf = calculate_metrics(y_train, y_test, y_pred_tr, y_pred_te)

# 5. SVM
from sklearn.svm import SVC

svm = SVC(kernel="poly")
svm.fit(x_train, y_train)

y_pred_tr = svm.predict(x_train)
y_pred_te = svm.predict(x_test)

# acc_train_svm, acc_test_svm, prec_svm, rec_svm = calculate_metrics(y_train, y_test, y_pred_tr, y_pred_te)

# 6. Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)

y_pred_tr = lr.predict(x_train)
y_pred_te = lr.predict(x_test)

# acc_train_lr, acc_test_lr, prec_lr, rec_lr = calculate_metrics(y_train, y_test, y_pred_tr, y_pred_te)

# 7. ANN
from sklearn.neural_network import MLPClassifier

ann = MLPClassifier(hidden_layer_sizes=256, activation="relu", solver="adam", batch_size="auto")

ann.fit(x_train, y_train)

y_pred_tr = ann.predict(x_train)
y_pred_te = ann.predict(x_test)

acc_train_ann, acc_test_ann, prec_ann, rec_ann = calculate_metrics(y_train, y_test, y_pred_tr, y_pred_te)