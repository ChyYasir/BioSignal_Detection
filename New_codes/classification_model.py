import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle


def train_on_svm(X_resampled, y_resampled, X_test, y_test_original, y_test_binarized):
    cv_svm = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=3, cv=cv_svm)
    grid.fit(X_resampled, y_resampled)

    print(grid.best_params_)
    print(grid.best_estimator_)

    y_pred = grid.predict(X_test)
    # Use the original y_test here for accuracy calculation
    accuracy = accuracy_score(y_test_original, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("Classification Report:")
    print(classification_report(y_test_original, y_pred, zero_division=1))

    conf_matrix = confusion_matrix(y_test_original, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Assuming you have already binarized y_test for ROC calculation
    n_classes = y_test_binarized.shape[1]
    y_score_svm = grid.decision_function(X_test)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score_svm[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    class_labels = ["Cesarean vs Rest", "Induced-Cesarean vs Rest", "Spontaneous vs Rest"]
    plt.figure()
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'ROC curve of {class_labels[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC - SVM')
    plt.legend(loc="lower right")
    plt.show()


def train_on_rf(X_resampled, y_resampled, X_test, y_test, X_train, y_train):
    print("For Random Forest")
    param_grid_rf = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_classifier = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=5)

    # Training the model as before
    rf_classifier.fit(X_resampled, y_resampled)

    # Making predictions
    y_pred_rf = rf_classifier.predict(X_test)

    # Getting the probability predictions for ROC calculations
    y_score_rf = rf_classifier.predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):  # Assuming n_classes is defined as the number of unique classes
        # Here, you ensure y_test_binarized and y_score_rf are correctly used
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score_rf[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    class_labels = ["Cesarean vs Rest", "Induced-Cesarean vs Rest", "Spontaneous vs Rest"]
    # Plotting
    plt.figure()
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'ROC curve of {class_labels[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC - Random Forest')
    plt.legend(loc="lower right")
    plt.show()


column_headers = ["area", "perimeter", "circularity", "variance", "bending_energy", "energy", "crest_factor", "mean_frequency", "median_frequency", "peak_to_peak_amplitude", "contraction_intensity", "contraction_power", "shannon_entropy", "sample_entropy", "Dispersion_entropy", "log_detector", "label"]

# Reload the datasets with the correct headers
cesarean_df = pd.read_csv('early_cesarean_features.csv', names=column_headers, header=None)
induced_cesarean_df = pd.read_csv('early_induced-cesarean_features.csv', names=column_headers, header=None)
spontaneous_df = pd.read_csv('early_spontaneous_features.csv', names=column_headers, header=None)

# Assigning labels
cesarean_df['label'] = 0
induced_cesarean_df['label'] = 1
spontaneous_df['label'] = 2

# Combine the datasets
combined_df = pd.concat([cesarean_df, induced_cesarean_df, spontaneous_df])
selected_columns = ["area", "perimeter", "variance", "bending_energy", "energy",
                    "peak_to_peak_amplitude", "contraction_power", "shannon_entropy", "log_detector"]


# "area", "perimeter", "variance", "bending_energy", "energy", "peak_to_peak_amplitude", "contraction_power", "shannon_entropy", "log_detector"

# Select the data for these columns
X = combined_df[selected_columns]
classes = [0, 1, 2]
y = combined_df['label']
# n_classes = y.shape[1]

# Splitting the dataset into the Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# train_on_rf(X_resampled, y_resampled, X_test, y_test)
# train_on_svm(X_resampled, y_resampled, X_test, y_test)

# Combine only Cesarean and Induced Cesarean datasets
# binary_df_1 = pd.concat([cesarean_df, spontaneous_df])
#
# # Select the data for columns
# X_binary_1 = binary_df_1[selected_columns]
# y_binary_1 = binary_df_1['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)


# Apply SMOTE on the training set
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Convert y_train and y_test for ROC plotting (after splitting to ensure matching shapes)
y_train_binarized = label_binarize(y_train, classes=[0, 1, 2])
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_train_binarized.shape[1]

# Now train your models
train_on_rf(X_resampled, y_resampled, X_test, y_test_binarized, X_train, y_train_binarized)
train_on_svm(X_resampled, y_resampled, X_test, y_test, y_test_binarized)

