import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.interpolate import interp1d


def smooth_curve(fpr, tpr):

    fpr_new = np.linspace(fpr.min(), fpr.max(), 1000000)
    tpr_interpolated = interp1d(fpr, tpr, kind='linear')(fpr_new)
    return fpr_new, tpr_interpolated


def plot_roc_curve(fpr, tpr, roc_auc, class_labels):
    plt.figure()
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

    for i, color in zip(range(n_classes), colors):
        fpr_smoothed, tpr_smoothed = smooth_curve(fpr[i], tpr[i])
        plt.plot(fpr_smoothed, tpr_smoothed, color=color, lw=lw,
                 label=f'ROC curve of {class_labels[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC with Interpolation')
    plt.legend(loc="lower right")
    plt.show()


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
    accuracy = accuracy_score(y_test_original, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("Classification Report:")
    print(classification_report(y_test_original, y_pred, zero_division=1))

    conf_matrix = confusion_matrix(y_test_original, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    n_classes = y_test_binarized.shape[1]
    y_score_svm = grid.decision_function(X_test)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score_svm[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    class_labels = ["Cesarean vs Rest", "Induced-Cesarean vs Rest", "Spontaneous vs Rest"]
    plot_roc_curve(fpr, tpr, roc_auc, class_labels)


def train_on_rf(X_resampled, y_resampled, X_test, y_test, X_train, y_train):
    print("For Random Forest")
    param_grid_rf = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_classifier = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=5)

    rf_classifier.fit(X_resampled, y_resampled)

    y_pred_rf = rf_classifier.predict(X_test)
    y_score_rf = rf_classifier.predict_proba(X_test)

    n_classes = y_train.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score_rf[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    class_labels = ["Cesarean vs Rest", "Induced-Cesarean vs Rest", "Spontaneous vs Rest"]
    plot_roc_curve(fpr, tpr, roc_auc, class_labels)


column_headers = ["area", "perimeter", "circularity", "variance", "bending_energy", "energy", "crest_factor",
                  "mean_frequency", "median_frequency", "peak_to_peak_amplitude", "contraction_intensity",
                  "contraction_power", "shannon_entropy", "sample_entropy", "Dispersion_entropy", "log_detector",
                  "label"]

cesarean_df = pd.read_csv('early_cesarean_features.csv')
induced_cesarean_df = pd.read_csv('early_induced-cesarean_features.csv')
spontaneous_df = pd.read_csv('early_spontaneous_features.csv')

cesarean_df['label'] = 0
induced_cesarean_df['label'] = 1
spontaneous_df['label'] = 2

combined_df = pd.concat([cesarean_df, induced_cesarean_df, spontaneous_df])
selected_columns = ["area", "perimeter", "variance", "bending_energy", "energy",
                    "peak_to_peak_amplitude", "contraction_power", "shannon_entropy", "log_detector"]

X = combined_df[selected_columns]
y = combined_df['label']

smote = SMOTE(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

y_train_binarized = label_binarize(y_train, classes=[0, 1, 2])
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_train_binarized.shape[1]

train_on_rf(X_resampled, y_resampled, X_test, y_test_binarized, X_train, y_train_binarized)
train_on_svm(X_resampled, y_resampled, X_test, y_test, y_test_binarized)
