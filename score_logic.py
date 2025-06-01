import pandas as pd
import numpy as np
import joblib # type: ignore
import json
import logging
import os
from typing import List, Dict, Any, Tuple, Optional, Union

# Conditional imports for optional libraries
try:
    from sklearn.calibration import CalibratedClassifierCV
    # Importer d'autres classes de modèles si nécessaire pour la détection de type
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    # Tenter d'importer les classifieurs optionnels pour la détection de type
    try:
        from lightgbm import LGBMClassifier # type: ignore
    except ImportError:
        LGBMClassifier = None # type: ignore
    try:
        from xgboost import XGBClassifier # type: ignore
    except ImportError:
        XGBClassifier = None # type: ignore
except ImportError:
    # Gérer le cas où scikit-learn n'est pas installé, bien que peu probable si le reste fonctionne
    logging.error("Scikit-learn n'est pas installé, certaines fonctionnalités de chargement de modèle peuvent échouer.")
    CalibratedClassifierCV = None # type: ignore
    RandomForestClassifier = None # type: ignore
    LogisticRegression = None # type: ignore
    LGBMClassifier = None # type: ignore
    XGBClassifier = None # type: ignore


# --- Configurations (Peuvent être externalisées plus tard) ---
# Ces chemins devront être ajustés ou rendus relatifs pour le déploiement web
FILE_PATHS: Dict[str, str] = {
    "model_filename": "aki_ultimate_ctk_model_v21_final_fixes.joblib", 
    "preprocessor_filename": "aki_ultimate_ctk_preprocessor_v21_final_fixes.joblib",
    "feature_selector_filename": "aki_ultimate_ctk_feature_selector_v21.joblib", 
    "metadata_filename": "aki_ultimate_ctk_metadata_v21_final_fixes.json",
    "shap_background_filename": "shap_background_data_v21_final_fixes.joblib",
}

# Configuration du logging (peut être configuré par l'application appelante)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_numpy_types_to_python(item: Any) -> Any:
    """Convertit les types NumPy en types Python natifs pour la sérialisation JSON."""
    if isinstance(item, dict):
        return {key: convert_numpy_types_to_python(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [convert_numpy_types_to_python(element) for element in item]
    elif isinstance(item, np.integer):
        return int(item)
    elif isinstance(item, np.floating):
        return float(item)
    elif isinstance(item, np.ndarray):
        return item.tolist()
    elif isinstance(item, np.bool_):
        return bool(item)
    return item

def load_model_data() -> Tuple[Optional[Any], Optional[Any], Optional[Any], 
                               Optional[List[str]], Optional[Dict[str, Any]], Optional[str], 
                               Optional[pd.Series], Optional[List[str]], Optional[List[str]], 
                               Optional[pd.DataFrame], Optional[float], Optional[bool]]:
    """
    Charge le préprocesseur, le pipeline modèle complet, le sélecteur de caractéristiques (si existant),
    et les métadonnées nécessaires à la prédiction.
    
    Retourne:
        Tuple contenant les objets chargés et les métadonnées, ou des Nones en cas d'échec.
    """
    logging.info("\n--- Chargement des Données du Modèle Sauvegardé (Logique) ---")
    
    # Vérifier l'existence des fichiers essentiels
    required_files_check = [
        FILE_PATHS['preprocessor_filename'], 
        FILE_PATHS['model_filename'], 
        FILE_PATHS['metadata_filename']
    ]
    if not all(os.path.exists(f) for f in required_files_check):
        logging.error("Un ou plusieurs fichiers de modèle/préprocesseur/métadonnées essentiels non trouvés.")
        return None, None, None, None, None, None, None, None, None, None, None, None

    try:
        preprocessor_loaded = joblib.load(FILE_PATHS['preprocessor_filename'])
        model_pipeline_loaded = joblib.load(FILE_PATHS['model_filename'])
        
        feature_selector_loaded = None
        if os.path.exists(FILE_PATHS['feature_selector_filename']):
            try:
                feature_selector_loaded = joblib.load(FILE_PATHS['feature_selector_filename'])
                logging.info(f"Sélecteur de caractéristiques chargé depuis {FILE_PATHS['feature_selector_filename']}")
            except Exception as e_fs_load:
                logging.warning(f"Impossible de charger le sélecteur de caractéristiques: {e_fs_load}")
        
        with open(FILE_PATHS['metadata_filename'], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        model_type_str = "Unknown"
        base_estimator_for_type_check = None

        # Essayer d'extraire le classifieur final pour déterminer son type
        current_estimator = model_pipeline_loaded
        if hasattr(current_estimator, 'named_steps'): # Si c'est un pipeline
            if 'calibrated_classifier' in current_estimator.named_steps and \
               hasattr(current_estimator.named_steps['calibrated_classifier'], 'base_estimator'):
                current_estimator = current_estimator.named_steps['calibrated_classifier'].base_estimator
            if 'classifier' in current_estimator.named_steps: # Peut être imbriqué (ex: SMOTE -> Classifier)
                base_estimator_for_type_check = current_estimator.named_steps['classifier']
            else: # Si le dernier pas du pipeline est le classifieur lui-même (après calibration par ex.)
                base_estimator_for_type_check = current_estimator 
        elif CalibratedClassifierCV and isinstance(current_estimator, CalibratedClassifierCV):
             base_estimator_for_type_check = current_estimator.base_estimator


        suffix = ""
        if hasattr(model_pipeline_loaded, 'named_steps') and 'calibrated_classifier' in model_pipeline_loaded.named_steps:
            suffix = " (Calibré)"
        elif CalibratedClassifierCV and isinstance(model_pipeline_loaded, CalibratedClassifierCV):
            suffix = " (Calibré)"


        if base_estimator_for_type_check:
            if RandomForestClassifier and isinstance(base_estimator_for_type_check, RandomForestClassifier): model_type_str = f"RandomForest{suffix}"
            elif LogisticRegression and isinstance(base_estimator_for_type_check, LogisticRegression): model_type_str = f"LogisticRegression{suffix}"
            elif LGBMClassifier is not None and isinstance(base_estimator_for_type_check, LGBMClassifier): model_type_str = f"LightGBM{suffix}"
            elif XGBClassifier is not None and isinstance(base_estimator_for_type_check, XGBClassifier): model_type_str = f"XGBoost{suffix}"
            elif suffix: model_type_str = f"Type Inconnu{suffix}"
            else: model_type_str = "Type Inconnu"
        elif suffix:  # Si c'est calibré mais le type de base n'est pas reconnu
            model_type_str = f"Type Inconnu{suffix}"


        X_train_dtypes_loaded_dict = metadata.get('X_train_dtypes')
        X_train_dtypes_loaded = pd.Series(X_train_dtypes_loaded_dict).astype(str) if X_train_dtypes_loaded_dict else None
        
        selected_features_initial_loaded = metadata.get('selected_features_initial')
        processed_feature_names_after_preproc_loaded = metadata.get('processed_feature_names_after_preproc')
        final_selected_feature_names_loaded = metadata.get('final_selected_feature_names') 
        optimal_threshold_youden_loaded = metadata.get('optimal_threshold_youden', 0.5) 
        feature_selection_applied_loaded = metadata.get('feature_selection_applied', False)
        
        shap_background_loaded = None
        if os.path.exists(FILE_PATHS['shap_background_filename']):
            try:
                shap_background_loaded = joblib.load(FILE_PATHS['shap_background_filename'])
                logging.info(f"Données de fond SHAP chargées depuis {FILE_PATHS['shap_background_filename']}")
            except Exception as e_shap:
                logging.error(f"Erreur lors du chargement des données de fond SHAP: {e_shap}", exc_info=True)

        logging.info(f"Données du modèle chargées avec succès. Type de modèle détecté: {model_type_str}. Seuil optimal Youden J: {optimal_threshold_youden_loaded:.4f}")
        logging.info(f"Sélection de caractéristiques appliquée lors de l'entraînement: {feature_selection_applied_loaded}")
        if final_selected_feature_names_loaded:
            logging.info(f"Nombre de caractéristiques finales sélectionnées: {len(final_selected_feature_names_loaded)}")

        return (preprocessor_loaded, model_pipeline_loaded, feature_selector_loaded,
                selected_features_initial_loaded, metadata.get('feature_value_maps'), model_type_str, 
                X_train_dtypes_loaded, processed_feature_names_after_preproc_loaded, final_selected_feature_names_loaded,
                shap_background_loaded, optimal_threshold_youden_loaded, feature_selection_applied_loaded) 
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données du modèle sauvegardé: {e}", exc_info=True)
        return None, None, None, None, None, None, None, None, None, None, None, None

def predict_new_patient(patient_data_dict: Dict[str, Any],
                        original_feature_columns: List[str], 
                        X_train_dtypes_original: Optional[pd.Series], 
                        preprocessor_for_shap: Any, # Peut être ColumnTransformer ou partie du pipeline
                        model_pipeline: Any, 
                        optimal_threshold: float, 
                        explainer: Optional[Any] = None, 
                        feature_names_for_shap: Optional[List[str]] = None,
                        feature_selection_applied: bool = False, # Indique si la sélection a été faite
                        feature_selector_fitted: Optional[Any] = None # Sélecteur fitté
                        ) -> Tuple[Optional[float], Optional[int], Optional[pd.DataFrame]]: 
    """
    Prédit le risque pour un nouveau patient.
    model_pipeline: Le pipeline complet (préprocesseur, sélecteur?, smote?, classifieur, calibrateur?).
    preprocessor_for_shap: Le préprocesseur fitté (utilisé pour transformer les données pour SHAP).
    feature_names_for_shap: Les noms des caractéristiques attendus par l'explainer SHAP.
    """
    logging.info("\n--- Prédiction pour Nouveau Patient (Logique) ---")
    try:
        new_patient_df = pd.DataFrame(columns=original_feature_columns, index=[0])
        for col in original_feature_columns:
            value = patient_data_dict.get(col)
            if isinstance(value, str) and value.strip() == '': 
                value = np.nan
            new_patient_df.loc[0, col] = value

        if X_train_dtypes_original is not None:
            for col in original_feature_columns:
                if col not in X_train_dtypes_original.index:
                    continue
                expected_dtype_str = str(X_train_dtypes_original[col])
                current_value = new_patient_df.loc[0, col]
                if pd.isna(current_value): continue
                if 'float' in expected_dtype_str or 'int' in expected_dtype_str:
                    try:
                        new_patient_df.loc[0, col] = pd.to_numeric(str(current_value).replace(',', '.'), errors='raise')
                    except (ValueError, TypeError):
                        new_patient_df.loc[0, col] = np.nan
                elif 'object' in expected_dtype_str or 'category' in expected_dtype_str:
                    new_patient_df.loc[0, col] = str(current_value)
        
        # Le pipeline complet gère le prétraitement, la sélection, etc.
        aki_probability_score = model_pipeline.predict_proba(new_patient_df)[0][1] 
        aki_prediction_binary = (aki_probability_score >= optimal_threshold).astype(int) 
        
        logging.info(f"Prédiction de probabilité IRA réussie. Score: {aki_probability_score:.4f}")
        logging.info(f"Prédiction binaire IRA (seuil {optimal_threshold:.4f}): {'Oui' if aki_prediction_binary == 1 else 'Non'}")

        shap_df_out = None
        # SHAP_AVAILABLE et shap sont vérifiés globalement, explainer est passé
        if explainer is not None and feature_names_for_shap is not None:
            try:
                logging.info("Calcul des valeurs SHAP pour la prédiction...")
                # 1. Transformer avec le préprocesseur seul
                new_patient_processed_df = preprocessor_for_shap.transform(new_patient_df)
                
                # 2. Si la sélection de caractéristiques a été appliquée lors de l'entraînement de l'explainer SHAP,
                #    il faut s'assurer que new_patient_processed_df a les bonnes colonnes.
                #    feature_names_for_shap devrait être la liste des caractéristiques APRÈS sélection.
                if feature_selection_applied and feature_selector_fitted is not None:
                    # On pourrait utiliser feature_selector_fitted.transform() mais il faut être sûr
                    # que les noms de colonnes correspondent. Le plus sûr est de se baser sur
                    # feature_names_for_shap qui sont les noms attendus par l'explainer.
                    if all(fn in new_patient_processed_df.columns for fn in feature_names_for_shap):
                        new_patient_processed_df = new_patient_processed_df[feature_names_for_shap]
                    else:
                        # Tenter de reconstruire avec les noms du préprocesseur si le sélecteur n'est pas directement applicable
                        # Cela suppose que feature_names_for_shap sont un sous-ensemble des colonnes du préprocesseur
                        temp_df_for_shap_cols = pd.DataFrame(columns=preprocessor_for_shap.get_feature_names_out())
                        if all(fn in temp_df_for_shap_cols.columns for fn in feature_names_for_shap):
                             new_patient_processed_df = new_patient_processed_df[feature_names_for_shap]
                        else:
                             raise ValueError(f"Les noms de caractéristiques pour SHAP ({feature_names_for_shap}) "
                                              f"ne sont pas un sous-ensemble des colonnes après préprocessing ({new_patient_processed_df.columns.tolist()}).")

                elif len(new_patient_processed_df.columns) != len(feature_names_for_shap):
                     raise ValueError(f"Discordance du nombre de colonnes pour SHAP. Attendu: {len(feature_names_for_shap)}, Obtenu: {len(new_patient_processed_df.columns)}")
                
                # S'assurer que les colonnes du DataFrame correspondent aux noms attendus par SHAP
                new_patient_processed_df.columns = feature_names_for_shap
                
                shap_values_raw = explainer.shap_values(new_patient_processed_df) 
                
                shap_values_for_prediction: np.ndarray
                # ... (logique d'extraction des shap_values, inchangée) ...
                if isinstance(explainer, shap.TreeExplainer):
                    if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2: 
                        shap_values_for_prediction = shap_values_raw[1][0] 
                    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 2: 
                         shap_values_for_prediction = shap_values_raw[0] 
                    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 1: 
                        shap_values_for_prediction = shap_values_raw
                    else: 
                        logging.warning(f"Format inattendu des valeurs SHAP de TreeExplainer. Type: {type(shap_values_raw)}. Tentative de fallback.")
                        shap_values_for_prediction = shap_values_raw[0] if isinstance(shap_values_raw, (list, np.ndarray)) and len(shap_values_raw) > 0 else np.array([])
                elif isinstance(explainer, shap.KernelExplainer):
                     if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
                        shap_values_for_prediction = shap_values_raw[1][0] 
                     elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 2: 
                         shap_values_for_prediction = shap_values_raw[0] 
                     else: 
                        logging.warning(f"Format inattendu des valeurs SHAP de KernelExplainer. Type: {type(shap_values_raw)}. Tentative de fallback.")
                        shap_values_for_prediction = shap_values_raw[0] if isinstance(shap_values_raw, (list, np.ndarray)) and len(shap_values_raw) > 0 else np.array([])
                else: 
                    logging.warning(f"Type d'explainer SHAP ('{type(explainer)}') non géré explicitement pour l'extraction des valeurs. Tentative de fallback.")
                    shap_values_for_prediction = shap_values_raw[0] if isinstance(shap_values_raw, list) else shap_values_raw


                if len(feature_names_for_shap) != len(shap_values_for_prediction):
                     raise ValueError(f"Le nombre de valeurs SHAP ({len(shap_values_for_prediction)}) ne correspond pas "
                                      f"au nombre de caractéristiques pour SHAP ({len(feature_names_for_shap)}) après extraction.")

                shap_df = pd.DataFrame({'Feature': feature_names_for_shap,'SHAP Value': shap_values_for_prediction})
                shap_df['Abs_SHAP'] = shap_df['SHAP Value'].abs() 
                shap_df_out = shap_df.sort_values(by='Abs_SHAP', ascending=False)
                logging.info("Valeurs SHAP calculées avec succès.")
            except Exception as shap_e:
                logging.error(f"--- ERREUR LORS DU CALCUL SHAP: {shap_e} ---", exc_info=True)
                shap_df_out = None 
        return aki_probability_score, aki_prediction_binary, shap_df_out 
    except Exception as e:
        logging.error(f"--- ERREUR LORS DE LA PRÉDICTION DU PATIENT: {e} ---", exc_info=True)
        return None, None, None 

# Les fonctions d'entraînement et d'évaluation (train_and_tune_model, evaluate_model, etc.)
# resteraient dans le script principal si l'entraînement est déclenché depuis là,
# ou pourraient aussi être déplacées dans score_logic.py si l'application web
# devait également gérer l'entraînement (ce qui est moins courant pour une simple app de prédiction).
# Pour l'instant, on se concentre sur la prédiction.
