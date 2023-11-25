import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
# Load term features
term_data = np.loadtxt('new_term_features.txt', max_rows=500, usecols=( 0, 4, 5))
term_labels = np.ones((term_data.shape[0], 1))  # Label for term features is 1
# print(term_data[0])
# Load preterm features
preterm_data = np.loadtxt('new_preterm_features.txt', max_rows=100, usecols=(0, 4, 5))
preterm_labels = np.zeros((preterm_data.shape[0], 1))  # Label for preterm features is 0

# Combine term and preterm data
data = np.vstack((term_data, preterm_data))
labels = np.vstack((term_labels, preterm_labels)).flatten()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

cnt_z = 0
cnt_o = 0

for i in y_train:
    if i == 0:
        cnt_z += 1
    else:
        cnt_o += 1
print("Before SMOTE: ", cnt_o, " , ", cnt_z)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
cnt_z = 0
cnt_o = 0

for i in y_resampled:
    if i == 0:
        cnt_z += 1
    else:
        cnt_o += 1
print("After SMOTE: ", cnt_o, ", ",  cnt_z)
# Initialize SVM classifier
svm_classifier = SVC(kernel='rbf', C=100, gamma=50)

# Train the classifier
svm_classifier.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


print("For Random Forest")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))