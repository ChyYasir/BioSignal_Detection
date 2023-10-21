import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split


term_features = np.loadtxt("term_features.txt", delimiter="\t")
preterm_features = np.loadtxt("preterm_features.txt", delimiter="\t")

term_features_list = term_features.tolist()
preterm_features_list = preterm_features.tolist()
all_features = term_features_list
print(term_features_list)


for i in range(len(preterm_features_list)):
    all_features.append(preterm_features_list[i])

X = np.array(all_features)
y = []
for i in range(24):
    if i <= 11:
        y.append(1) #for term
    else:
        y.append(0) #for preterm
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# lr = LogisticRegression(max_iter=1000, random_state=42)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_test_predicted = lr.predict(X_test)
print(y_test_predicted)

cm = confusion_matrix(y_test, y_test_predicted, labels=lr.classes_)
print(cm)

cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["preterm", "term"])
cm_disp.plot()
plt.show()

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

