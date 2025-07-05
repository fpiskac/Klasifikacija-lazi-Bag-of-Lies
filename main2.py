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

# Putanja do organiziranih EEG tablica
organized_path = r"C:\\OrganizedFolders"

# Učitavanje klasa iz Annotations.csv
annotations_file = r"Annotations.csv"
annotations = pd.read_csv(annotations_file)
classes = annotations['truth'].tolist()

# Dohvat svih foldera
folders = [os.path.join(organized_path, f) for f in os.listdir(organized_path)
           if os.path.isdir(os.path.join(organized_path, f))]

# Kolone koje nas zanimaju
required_columns_eeg = ['F3 Value', 'F4 Value', 'FC5 Value', 'F7 Value', 'AF4 Value', 'FC6 Value']
required_columns_gaze = ['FPOGD', 'CS', 'LPD', 'RPD']

# Za spremanje sažetih značajki
summary_rows = []
class_index = 0

for folder in folders:
    eeg_file = os.path.join(folder, 'EEG.csv')
    gaze_file = os.path.join(folder, 'Gaze.csv')

    # Učitavanje podataka
    if not os.path.exists(eeg_file) or not os.path.exists(gaze_file):
        print(f"Skipping {folder} - nedostaje EEG ili Gaze datoteka.")
        continue

    eeg_data = pd.read_csv(eeg_file)
    gaze_data = pd.read_csv(gaze_file)

    # Provjera potrebnih kolona
    if not all(col in eeg_data.columns for col in required_columns_eeg):
        print(f"Skipping {folder} - EEG datoteka nema sve potrebne kolone.")
        continue
    if not all(col in gaze_data.columns for col in required_columns_gaze):
        print(f"Skipping {folder} - Gaze datoteka nema sve potrebne kolone.")
        continue

    # Kombiniranje i čišćenje podataka
    eeg_data = eeg_data[required_columns_eeg].apply(pd.to_numeric, errors='coerce')
    gaze_data = gaze_data[required_columns_gaze].apply(pd.to_numeric, errors='coerce')
    combined = pd.concat([eeg_data, gaze_data], axis=1)

    # Osiguravanje numeričkog tipa (izbjegava future warning)
    combined = combined.infer_objects(copy=False)

    # Interpolacija i popuna NaN vrijednosti
    combined = combined.interpolate(method='linear', limit_direction='both')
    combined = combined.fillna(combined.mean(numeric_only=True))

    # Sažetak (statističke značajke)
    summary = {}
    for col in combined.columns:
        summary[f"{col}_min"] = combined[col].min()
        summary[f"{col}_max"] = combined[col].max()
        summary[f"{col}_median"] = combined[col].median()
        summary[f"{col}_mean"] = combined[col].mean()
        summary[f"{col}_std"] = combined[col].std()

    # Dodavanje klase
    if class_index >= len(classes):
        print(f"Upozorenje: Nedostaje klasa za {folder}")
        continue
    summary["Class"] = classes[class_index]
    summary_rows.append(summary)
    class_index += 1

# Stvaranje DataFramea
summary_df = pd.DataFrame(summary_rows)

print("\nPregled tablice sa sažetim značajkama po pokušaju:")
print(summary_df.head())

# Razdvajanje X i y
X = summary_df.drop(columns=['Class'])
y = summary_df['Class']

# Definicija modela
models = {
    "SVM": make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=42)),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier())
}

# Metrike za evaluaciju
metrics = {
    'Accuracy': 'accuracy',
    'F1 Score': 'f1',
    'ROC AUC': 'roc_auc'
}

# Križna validacija
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
results = {}

for name, model in models.items():
    cv_results = cross_validate(model, X, y, cv=kf, scoring=metrics, n_jobs=-1)
    results[name] = {
        'Accuracy': f"{cv_results['test_Accuracy'].mean():.3f} ± {cv_results['test_Accuracy'].std():.3f}",
        'F1 Score': f"{cv_results['test_F1 Score'].mean():.3f} ± {cv_results['test_F1 Score'].std():.3f}",
        'ROC AUC': f"{cv_results['test_ROC AUC'].mean():.3f} ± {cv_results['test_ROC AUC'].std():.3f}"
    }

# Ispis rezultata
results_df = pd.DataFrame(results).T
print("\nRezultati evaluacije:")
print(results_df)
