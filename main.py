import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Path to organized EEG tables
organized_path = r"C:\\OrganizedFolders"

# Loading classes (truth/lie) from Annotations.csv
annotations_file = r"Annotations.csv"
annotations = pd.read_csv(annotations_file)
classes = annotations['truth'].tolist()
print(classes)

# Finding all folders in the organized directory
folders = [os.path.join(organized_path, f) for f in os.listdir(organized_path) if
           os.path.isdir(os.path.join(organized_path, f))]

# Empty DataFrame for merging all tables
combined_data = pd.DataFrame()

# Index tracking the classes
class_index = 0

# Iteratively processing each EEG.csv and Gaze.csv file within folders
for folder in folders:
    eeg_file = os.path.join(folder, 'EEG.csv')
    gaze_file = os.path.join(folder, 'Gaze.csv')

    eeg_data = None
    gaze_data = None

    # Checking if the EEG.csv file exists
    if os.path.exists(eeg_file):
        eeg_data = pd.read_csv(eeg_file)
        required_columns_eeg = ['F3 Value', 'F4 Value', 'FC5 Value', 'F7 Value', 'AF4 Value', 'FC6 Value']
        missing_columns_eeg = [col for col in required_columns_eeg if col not in eeg_data.columns]
        if missing_columns_eeg:
            print(f"Warning: File {eeg_file} is missing columns {missing_columns_eeg}. Skipping this file.")
            eeg_data = None

    # Checking if the Gaze.csv file exists
    if os.path.exists(gaze_file):
        gaze_data = pd.read_csv(gaze_file)
        required_columns_gaze = ['FPOGD', 'CS', 'LPD', 'RPD']
        missing_columns_gaze = [col for col in required_columns_gaze if col not in gaze_data.columns]
        if missing_columns_gaze:
            print(f"Warning: File {gaze_file} is missing columns {missing_columns_gaze}. Skipping this file.")
            gaze_data = None

    # If both EEG and Gaze data are not loaded, skip this folder
    if eeg_data is None or gaze_data is None:
        print(f"Skipping folder '{folder}' as either EEG or Gaze data is missing.")
        continue

    # Retaining and transposing EEG data
    eeg_data = eeg_data[required_columns_eeg].transpose()
    eeg_data = eeg_data.iloc[:, :600]  # Trimming to the first 600 columns

    # Retaining relevant columns from Gaze data
    gaze_data = gaze_data[required_columns_gaze].transpose()
    gaze_data = gaze_data.iloc[:, :600]  # Trimming to the first 600 columns

    max_columns = 600
    if gaze_data.shape[1] < max_columns:
        missing_cols = max_columns - gaze_data.shape[1]
        # Creating a DataFrame with NaN values and appropriate column names
        missing_data = pd.DataFrame(np.nan, index=gaze_data.index,
                                    columns=range(gaze_data.shape[1], gaze_data.shape[1] + missing_cols))
        gaze_data = pd.concat([gaze_data, missing_data], axis=1)
    if eeg_data.shape[1] < max_columns:
        missing_cols = max_columns - eeg_data.shape[1]
        # Creating a DataFrame with NaN values and appropriate column names
        missing_data = pd.DataFrame(np.nan, index=eeg_data.index,
                                    columns=range(eeg_data.shape[1], eeg_data.shape[1] + missing_cols))
        eeg_data = pd.concat([eeg_data, missing_data], axis=1)

    # Combining EEG and Gaze data
    combined_features = pd.concat([eeg_data.reset_index(drop=True), gaze_data], axis=1)

    # Adding the 'Class' column with the corresponding class
    combined_features['Class'] = classes[class_index]  # Adding class from annotations

    # Adding to the combined DataFrame
    combined_data = pd.concat([combined_data, combined_features], ignore_index=True)

    # Incrementing the class index
    class_index += 1

# Checking for NaN values in the data before imputation
if combined_data.isnull().values.any():
    print("Warning: There are NaN values in the data! Applying imputation.")
else:
    print("No NaN values found in the data.")

# First, remove the 'Class' column from imputation, as it is a categorical column.
numeric_data = combined_data.drop(columns=['Class'])

# Converting all data to numeric type (if not already)
numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')

# Interpolation to fill data based on existing values
numeric_data_imputed = numeric_data.interpolate(method='linear', limit_direction='both')

# If there are still NaN values (e.g., if all data in a column are NaN), fill them with the mean
numeric_data_imputed = numeric_data_imputed.fillna(numeric_data_imputed.mean())

# Adding the 'Class' column back to the imputed data
combined_data_imputed = pd.concat([numeric_data_imputed, combined_data['Class']], axis=1)

print(combined_data_imputed)

# Printing the total number of rows after imputation
print(f"Total number of rows in the combined table after imputation: {len(combined_data_imputed)}")

# Splitting the data
X = combined_data_imputed.drop(columns=['Class'])
y = combined_data_imputed['Class']

# Defining models and metrics
models = {
    "SVM": make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=42)),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier())
}

metrics = {
    'Accuracy': 'accuracy',
    'F1 Score': 'f1',
    'ROC AUC': 'roc_auc'
}

# Evaluation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
results = {}

for name, model in models.items():
    cv_results = cross_validate(model, X, y, cv=kf, scoring=metrics, n_jobs=-1)
    results[name] = {
        'Accuracy': f"{cv_results['test_Accuracy'].mean():.3f} ± {cv_results['test_Accuracy'].std():.3f}",
        'F1 Score': f"{cv_results['test_F1 Score'].mean():.3f} ± {cv_results['test_F1 Score'].std():.3f}",
        'ROC AUC': f"{cv_results['test_ROC AUC'].mean():.3f} ± {cv_results['test_ROC AUC'].std():.3f}"
    }

# Displaying results
results_df = pd.DataFrame(results).T
print("\nEvaluation results:")
print(results_df)