import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def train_on_svm(X_resampled, y_resampled, X_test, y_test):
    cv_svm = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=3, cv=cv_svm)

    # fitting the model for grid search
    grid.fit(X_resampled, y_resampled)
    # print best parameter after tuning
    print(grid.best_params_)
    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)
    y_pred = grid.predict(X_test)
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

    y_pred_proba = grid.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print("SVM AUC = ", roc_auc)
    return fpr, tpr, roc_auc
    # plt.figure()
    # plt.plot(fpr, tpr)
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristics - SVM")
    # plt.legend(loc="lower right")
    # plt.show()


def train_on_rf(X_resampled, y_resampled, X_test, y_test, X_train, y_train):
    print("For Random Forest")
    param_grid_rf = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_classifier = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Perform K-fold cross-validation on the training set
    cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    # Print cross-validation scores
    print("Cross-Validation Scores:", cv_scores)
    rf_classifier.fit(X_resampled, y_resampled)
    # Make predictions on the test set
    y_pred_rf = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_rf, zero_division=1))
    conf_matrix = confusion_matrix(y_test, y_pred_rf)
    print("Confusion Matrix:")
    print(conf_matrix)

    y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print("RF AUC = ", roc_auc)
    return fpr, tpr, roc_auc
    # plt.figure()
    # plt.plot(fpr, tpr)
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristics - RF")
    # plt.legend(loc="lower right")
    # plt.show()


def prepare_data_for_binary_classifcation(df, class_a, class_b):
    binary_df = df[df["label"].isin([class_a, class_b])]
    binary_df["label"] = np.where(binary_df["label"] == class_a, 0, 1)
    X = binary_df[selected_columns]
    y = binary_df["label"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def plot_roc_rf():
    for class_a, class_b, label in class_pairs:
        X_train, X_test, y_train, y_test = prepare_data_for_binary_classifcation(combined_df, class_a, class_b)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        fpr, tpr, roc_auc = train_on_rf(X_resampled, y_resampled, X_test, y_test, X_train, y_train)
        # train_on_svm(X_resampled, y_resampled, X_test, y_test)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    # X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_binary_1, y_binary_1, test_size=0.2, random_state=42 , stratify=y_binary_1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Comparisons (RF)')
    plt.legend(loc="lower right")
    plt.show()

def plot_roc_svm():
    for class_a, class_b, label in class_pairs:
        X_train, X_test, y_train, y_test = prepare_data_for_binary_classifcation(combined_df, class_a, class_b)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        fpr, tpr, roc_auc = train_on_svm(X_resampled, y_resampled, X_test, y_test)

        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    # X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_binary_1, y_binary_1, test_size=0.2, random_state=42 , stratify=y_binary_1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Comparisons (SVM)')
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
# Select the data for these columns
# X = combined_df[selected_columns]
# y= combined_df['label']


# Splitting the dataset into the Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# train_on_rf(X_resampled, y_resampled, X_test, y_test)
# train_on_svm(X_resampled, y_resampled, X_test, y_test)

# Combine only Cesarean and Induced Cesarean datasets
# binary_df_1 = pd.concat([cesarean_df, induced_cesarean_df])
#
# # Select the data for columns
# X_binary_1 = binary_df_1[selected_columns]
# y_binary_1 = binary_df_1['label']
class_pairs = [(0, 1, "Cesarean vs. Induced-Cesarean"),(1, 2, 'Induced-Cesarean vs. Spontaneous'),
                   (0, 2, 'Cesarean vs. Spontaneous') ]

plot_roc_rf()
plot_roc_svm()

# X_resampled_1, y_resampled_1 = smote.fit_resample(X_train_1, y_train_1)
#
#
# train_on_rf(X_resampled_1, y_resampled_1, X_test_1, y_test_1, X_train_1, y_train_1)
# train_on_svm(X_resampled_1, y_resampled_1, X_test_1, y_test_1)