import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


term_features = np.loadtxt("term_features.txt", delimiter="\t")
preterm_features = np.loadtxt("preterm_features.txt", delimiter="\t")

term_features_list = term_features.tolist()
preterm_features_list = preterm_features.tolist()


all_features = [item for item in term_features_list]

# print(term_features_list)


for i in range(len(preterm_features_list)):
    all_features.append(preterm_features_list[i])


# print(all_features)
# print(len(all_features))


X = np.array(all_features)
y = []
# print(len(term_features_list))
for i in range(len(term_features_list)):
    y.append(1)

for i in range(len(preterm_features_list)):
    y.append(0)

# print(len(y))
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Apply SMOTE to the training set
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(len(X_train_resampled))
cnt_term = 0
cnt_pre_term = 0

for i in range(len(y_train_resampled)):
    if y_train_resampled[i] == 0:
        cnt_pre_term += 1
    else:
        cnt_term += 1

print(cnt_term)
print(cnt_pre_term)
# lr = LogisticRegression(max_iter=1000, random_state=42)
lr = LogisticRegression()
lr.fit(X_train_resampled, y_train_resampled)
y_test_predicted = lr.predict(X_test)
# print(y_test_predicted)

cm = confusion_matrix(y_test, y_test_predicted, labels=lr.classes_)
print(cm)

cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["preterm", "term"])
cm_disp.plot()
plt.show()

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

