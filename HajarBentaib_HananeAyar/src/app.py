import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import os
import base64

# --- Page Config ---
st.set_page_config(
    page_title="Drug Discovery-AI | Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Rich Aesthetics) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    
    /* Main title styling */
    .title-text {
        background: linear-gradient(135deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Card style */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Status styling */
    .active-badge {
        background-color: #2e7d32;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.3);
    }
    
    .inactive-badge {
        background-color: #c62828;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(198, 40, 40, 0.3);
    }
    
    .info-box {
        background-color: rgba(28, 126, 214, 0.1);
        border-left: 5px solid #1c7ed6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.5rem;
    }
    
    </style>
""", unsafe_allow_html=True)

# --- Load Models & Scalers ---
@st.cache_resource
def load_assets():
    scaler_bace = joblib.load('models/scaler_bace.joblib')
    bace_clf = joblib.load('models/bace_classifier.joblib')
    scaler_davis = joblib.load('models/scaler_davis.joblib')
    davis_encoder = joblib.load('models/davis_target_encoder.joblib')
    davis_reg = joblib.load('models/davis_regressor.joblib')
    return scaler_bace, bace_clf, scaler_davis, davis_encoder, davis_reg

try:
    scaler_bace, bace_clf, scaler_davis, davis_encoder, davis_reg = load_assets()
    loaded_successfully = True
except Exception as e:
    st.error(f"Erreur lors du chargement des modèles : {str(e)}")
    loaded_successfully = False

# --- Helper Functions ---
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

def draw_mol_to_svg(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            d2d = rdMolDraw2D.MolDraw2DSVG(400, 400)
            d2d.DrawMolecule(mol)
            d2d.FinishDrawing()
            svg = d2d.GetDrawingText()
            b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
            return f"data:image/svg+xml;base64,{b64}"
    except:
        pass
    return None

# --- Sidebar Navigation ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/822/822143.png", width=80)
st.sidebar.markdown("<h2 style='margin-bottom: 0px;'>Drug Discovery-AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: gray; font-size: 0.9rem;'>Platforme d'IA pour la Bio-informatique</p>", unsafe_allow_html=True)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🔬 BACE-1 Predictor (Classification)", "🧬 DAVIS Affinity Predictor (Régression)", "📊 Explore Data & Stats"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    ### À propos des descripteurs
    - **MW (Molecular Weight)** : Masse moléculaire totale.
    - **LogP** : Lipophilie (coefficient de partage octanol/eau).
    - **TPSA** : Surface polaire topologique (pénétration membranaire).
    - **HBA/HBD** : Accepteurs/Donneurs de liaisons hydrogène.
""")

# --- PAGE 1: BACE-1 Predictor ---
if page == "🔬 BACE-1 Predictor (Classification)" and loaded_successfully:
    st.markdown("<div class='title-text'>🔬 BACE-1 Predictor</div>", unsafe_allow_html=True)
    st.write("Prédisez l'activité inhibitrice biologique d'une molécule contre la cible thérapeutique **BACE-1**.")

    st.markdown("""
        <div class='info-box'>
            Entrez la structure SMILES d'une molécule. L'application calculera automatiquement ses descripteurs 
            chimiques et utilisera notre modèle <b>Gradient Boosting</b> pour prédire son activité biologique (Active / Inactive).
        </div>
    """, unsafe_allow_html=True)

    # Examples
    st.markdown("**Molécules d'exemples :**")
    ex_cols = st.columns(3)
    ex1 = "O1CC[C@@H](NC(=O)[C@@H](Cc2cc3cc(ccc3nc2N)-c2ccccc2)Cc2ccccc2)C1" # Active
    ex2 = "Clc1cc2nc(n(c2cc1)CCCC(=O)NCC1CC1)N" # Inactive
    ex3 = "CC(=O)NC1=CC=C(O)C=C1" # Paracetamol (Inactive)

    if ex_cols[0].button("Molécule Active BACE-1 (Exemple 1)"):
        st.session_state.bace_smiles_input = ex1
    if ex_cols[1].button("Molécule Inactive BACE-1 (Exemple 2)"):
        st.session_state.bace_smiles_input = ex2
    if ex_cols[2].button("Paracétamol (Exemple 3)"):
        st.session_state.bace_smiles_input = ex3

    smiles_input = st.text_input(
        "Entrez une chaîne SMILES :",
        value=ex1,
        key="bace_smiles_input"
    )

    if smiles_input:
        col1, col2 = st.columns([1, 1])

        # Validate & Compute Descriptors
        desc = compute_descriptors(smiles_input)
        mol_svg = draw_mol_to_svg(smiles_input)

        with col1:
            st.subheader("Structure Chimique 2D")
            if mol_svg:
                st.image(mol_svg, caption="Générée avec RDKit", use_container_width=False, width=320)
            else:
                st.error("Structure SMILES invalide. Veuillez vérifier le format.")

        with col2:
            st.subheader("Descripteurs Physico-chimiques")
            if desc:
                desc_df = pd.DataFrame([desc])
                # Show key metrics in columns
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Masse Moléculaire (MW)", f"{desc['MW']:.2f}")
                m_col2.metric("Lipophilie (LogP)", f"{desc['LogP']:.2f}")
                m_col3.metric("Polarité (TPSA)", f"{desc['TPSA']:.2f}")
                
                st.dataframe(desc_df.T.rename(columns={0: 'Valeur'}))
            else:
                st.warning("Impossible de calculer les descripteurs.")

        if desc:
            st.markdown("---")
            st.subheader("Prédiction du Modèle d'IA")
            
            # Scale features
            features = pd.DataFrame([desc])
            X_scaled = scaler_bace.transform(features)
            
            # Predict
            pred = bace_clf.predict(X_scaled)[0]
            prob = bace_clf.predict_proba(X_scaled)[0]
            
            pred_col1, pred_col2 = st.columns(2)
            
            with pred_col1:
                st.markdown("<br>", unsafe_allow_html=True)
                if pred == 1:
                    st.markdown("<div class='active-badge'>✓ Molécule Active contre BACE-1</div>", unsafe_allow_html=True)
                    st.success(f"La molécule a été classée comme active avec une probabilité de {prob[1]*100:.2f}%.")
                else:
                    st.markdown("<div class='inactive-badge'>✗ Molécule Inactive contre BACE-1</div>", unsafe_allow_html=True)
                    st.error(f"La molécule a été classée comme inactive avec une probabilité de {prob[0]*100:.2f}%.")
            
            with pred_col2:
                # Gauge or probability distribution plot
                prob_df = pd.DataFrame({
                    'Classe': ['Inactive', 'Active'],
                    'Probabilité': prob
                })
                st.bar_chart(prob_df.set_index('Classe'))

# --- PAGE 2: DAVIS Predictor ---
elif page == "🧬 DAVIS Affinity Predictor (Régression)" and loaded_successfully:
    st.markdown("<div class='title-text'>🧬 DAVIS Affinity Predictor</div>", unsafe_allow_html=True)
    st.write("Estimez quantitativement l'affinité de liaison médicament-protéine ($K_d$ en nM).")

    st.markdown("""
        <div class='info-box'>
            Entrez la structure SMILES de la molécule d'intérêt et sélectionnez la protéine cible. Le modèle 
            <b>Gradient Boosting Regressor</b> prédira la constante de dissociation Kd.
            <i>Plus le Kd est petit, plus l'affinité de liaison est forte.</i>
        </div>
    """, unsafe_allow_html=True)

    # Target proteins
    targets = list(davis_encoder.classes_)
    
    # Input columns
    col_in1, col_in2 = st.columns([2, 1])
    
    with col_in1:
        davis_smiles = st.text_input(
            "Entrez une chaîne SMILES du Médicament :",
            value="Cc1[nH]nc2ccc(-c3cncc(OCC(N)Cc4ccccc4)c3)cc12"
        )
    with col_in2:
        target_name = st.selectbox(
            "Sélectionnez la Protéine Cible (Target ID) :",
            targets,
            index=targets.index("AAK1") if "AAK1" in targets else 0
        )

    if davis_smiles and target_name:
        col1, col2 = st.columns([1, 1])

        # Validate & Compute Descriptors
        desc = compute_descriptors(davis_smiles)
        mol_svg = draw_mol_to_svg(davis_smiles)

        with col1:
            st.subheader("Structure Chimique 2D")
            if mol_svg:
                st.image(mol_svg, caption="Générée avec RDKit", use_container_width=False, width=320)
            else:
                st.error("SMILES invalide.")

        with col2:
            st.subheader("Descripteurs Chimiques")
            if desc:
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Masse (MW)", f"{desc['MW']:.2f}")
                m_col2.metric("LogP", f"{desc['LogP']:.2f}")
                m_col3.metric("TPSA", f"{desc['TPSA']:.2f}")
                
                st.dataframe(pd.DataFrame([desc]).T.rename(columns={0: 'Valeur'}))
            else:
                st.warning("Impossible de calculer les descripteurs.")

        if desc:
            st.markdown("---")
            st.subheader("Prédiction d'Affinité Kd")
            
            # Preprocess features
            features = pd.DataFrame([desc])
            X_scaled = scaler_davis.transform(features)
            
            # Encode target
            target_encoded = davis_encoder.transform([target_name])[0]
            
            # Combine
            X_input = np.hstack([X_scaled, [[target_encoded]]])
            
            # Predict
            predicted_kd = davis_reg.predict(X_input)[0]
            
            # Show predictions
            pred_col1, pred_col2 = st.columns(2)
            
            with pred_col1:
                st.metric(
                    label=f"Affinité de liaison Kd prédite sur {target_name}",
                    value=f"{predicted_kd:.2f} nM",
                    delta="Forte affinité" if predicted_kd < 100 else "Faible affinité",
                    delta_color="normal" if predicted_kd < 100 else "inverse"
                )
                
                if predicted_kd < 10:
                    st.success("🔥 Liaison extrêmement forte (Sous-nanomolaire / Nanomolaire fort). Candidat prometteur !")
                elif predicted_kd < 100:
                    st.info("👍 Liaison modérée à forte. Activité biochimique significative.")
                else:
                    st.warning("⚠️ Liaison faible. La molécule a peu d'affinité pour cette cible.")

            with pred_col2:
                # Add some context about the drug discovery target
                st.markdown(f"""
                **Informations sur l'interaction :**
                - **Médicament** : `{davis_smiles[:40]}...`
                - **Cible** : `{target_name}`
                - **Encodage Cible** : `{target_encoded}`
                - **Constante d'inhibition** : $K_d$ sert d'indicateur pour évaluer l'efficacité de molécules thérapeutiques en phase de criblage.
                """)

# --- PAGE 3: Stats & Exploratory Data ---
elif page == "📊 Explore Data & Stats":
    st.markdown("<div class='title-text'>📊 Statistiques & Jeux de Données</div>", unsafe_allow_html=True)
    st.write("Explorez les dimensions et caractéristiques des jeux de données d'apprentissage propres.")

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Données BACE (Nettoyées)")
        if os.path.exists('Data/processed/bace_clean.csv'):
            df_b = pd.read_csv('Data/processed/bace_clean.csv')
            st.write(f"Dimensions : **{df_b.shape[0]}** molécules, **{df_b.shape[1]}** colonnes.")
            st.dataframe(df_b)
            
            # Class balance plot
            if 'Class' in df_b.columns:
                st.markdown("**Distribution des classes (BACE) :**")
                class_counts = df_b['Class'].value_counts()
                st.bar_chart(class_counts)
        else:
            st.warning("Fichier BACE propre non trouvé.")

    with col2:
        st.subheader("Données DAVIS (Nettoyées)")
        if os.path.exists('Data/processed/davis_clean.csv'):
            df_d = pd.read_csv('Data/processed/davis_clean.csv')
            st.write(f"Dimensions : **{df_d.shape[0]}** couples drogue-cible, **{df_d.shape[1]}** colonnes.")
            st.dataframe(df_d)
            
            # Targets count
            if 'Target_ID' in df_d.columns:
                st.markdown(f"Nombre de protéines cibles uniques : **{df_d['Target_ID'].nunique()}**")
            elif 'Target' in df_d.columns:
                st.markdown(f"Nombre de protéines cibles uniques : **{df_d['Target'].nunique()}**")
        else:
            st.warning("Fichier DAVIS propre non trouvé.")
