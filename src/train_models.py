import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from imblearn.over_sampling import SMOTE
import os

print("Starting model training script...")

# Create models directory
os.makedirs('models', exist_ok=True)

feature_cols = ['MW', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'RotatableBonds', 'RingCount', 'HeavyAtomCount']

def compute_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'RotatableBonds': Descriptors.NumRotatableBonds(mol),
            'RingCount': Descriptors.RingCount(mol),
            'HeavyAtomCount': Descriptors.HeavyAtomCount(mol)
        }
    except:
        return None

# --- Train BACE Classifier ---
print("Processing BACE dataset...")
df_bace = pd.read_csv('Data/bace.csv')

# Drop invalid molecules
df_bace['mol_obj'] = df_bace['mol'].apply(Chem.MolFromSmiles)
df_bace = df_bace.dropna(subset=['mol_obj']).reset_index(drop=True)

# Calculate descriptors
bace_desc = []
for s in df_bace['mol']:
    bace_desc.append(compute_descriptors(s))
df_bace_desc = pd.DataFrame(bace_desc)

# Fit and save BACE Scaler
scaler_bace = StandardScaler()
X_bace_scaled = scaler_bace.fit_transform(df_bace_desc[feature_cols])
joblib.dump(scaler_bace, 'models/scaler_bace.joblib')
print("BACE Scaler saved.")

# Balance using SMOTE
y_bace = df_bace['Class']
smote = SMOTE(random_state=42)
X_bace_res, y_bace_res = smote.fit_resample(X_bace_scaled, y_bace)

# Train Gradient Boosting Classifier
clf = GradientBoostingClassifier(random_state=42, n_estimators=100)
clf.fit(X_bace_res, y_bace_res)
joblib.dump(clf, 'models/bace_classifier.joblib')
print("BACE Classifier saved.")

# --- Train DAVIS Regressor ---
print("Processing DAVIS dataset...")
df_davis = pd.read_csv('Data/davis.csv')

# Drop invalid molecules
df_davis['mol_obj'] = df_davis['Drug'].apply(Chem.MolFromSmiles)
df_davis = df_davis.dropna(subset=['mol_obj']).reset_index(drop=True)

# Calculate unique drug descriptors to speed up computation
unique_drugs = df_davis['Drug'].unique()
print(f"Computing descriptors for {len(unique_drugs)} unique drugs in DAVIS...")
drug_desc_map = {}
for d in unique_drugs:
    desc = compute_descriptors(d)
    if desc is not None:
        drug_desc_map[d] = desc

# Map back to the dataframe
davis_desc = []
for d in df_davis['Drug']:
    davis_desc.append(drug_desc_map.get(d, compute_descriptors(d)))
df_davis_desc = pd.DataFrame(davis_desc)

# Fill any missing values with column means just in case
df_davis_desc = df_davis_desc.fillna(df_davis_desc.mean())

# Fit and save DAVIS Scaler
scaler_davis = StandardScaler()
X_davis_scaled = scaler_davis.fit_transform(df_davis_desc[feature_cols])
joblib.dump(scaler_davis, 'models/scaler_davis.joblib')
print("DAVIS Scaler saved.")

# Encode and save Target_ID
le_target = LabelEncoder()
df_davis['Target_ID_encoded'] = le_target.fit_transform(df_davis['Target_ID'])
joblib.dump(le_target, 'models/davis_target_encoder.joblib')
print("DAVIS Target Encoder saved.")

# Combine scaled descriptors and encoded target
X_davis = np.hstack([X_davis_scaled, df_davis[['Target_ID_encoded']].values])
y_davis = df_davis['Y']

# Train Gradient Boosting Regressor
reg = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
reg.fit(X_davis, y_davis)
joblib.dump(reg, 'models/davis_regressor.joblib')
print("DAVIS Regressor saved.")

print("All models and scalers have been trained and saved successfully in the 'models' directory!")
