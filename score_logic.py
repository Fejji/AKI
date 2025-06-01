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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectFromModel # Ajouté pour le type hint
    from sklearn.compose import ColumnTransformer # Ajouté pour le type hint
    from sklearn.pipeline import Pipeline # Ajouté pour le type hint
    try:
        from lightgbm import LGBMClassifier # type: ignore
    except ImportError:
        LGBMClassifier = None # type: ignore
    try:
        from xgboost import XGBClassifier # type: ignore
    except ImportError:
        XGBClassifier = None # type: ignore
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline # type: ignore
    except ImportError:
        ImbPipeline = None # type: ignore


except ImportError:
    logging.error("Scikit-learn ou imbalanced-learn n'est pas installé, certaines fonctionnalités de chargement de modèle peuvent échouer.")
    CalibratedClassifierCV = None # type: ignore
    RandomForestClassifier = None # type: ignore
    LogisticRegression = None # type: ignore
    LGBMClassifier = None # type: ignore
    XGBClassifier = None # type: ignore
    SelectFromModel = None # type: ignore
    ColumnTransformer = None # type: ignore
    Pipeline = None # type: ignore
    ImbPipeline = None # type: ignore


# --- Configurations ---
# MODIFIÉ: Utilisation de chemins relatifs (suppose que les fichiers sont à la racine du dépôt)
# Le fichier .sav n'est pas nécessaire pour la prédiction, seulement pour l'entraînement.
FILE_PATHS: Dict[str, str] = {
    # "sav_file": "Base.sav", # Non utilisé pour la prédiction
    "model_filename": "aki_ultimate_ctk_model.joblib", 
    "preprocessor_filename": "aki_ultimate_ctk_preprocessor.joblib",
    "feature_selector_filename": "aki_ultimate_ctk_feature_selector.joblib", 
    "metadata_filename": "aki_ultimate_ctk_metadata.json",
    "shap_background_filename": "shap_background_data.joblib",
}

# Configuration du logging
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

def load_model_data() -> Tuple[Optional[ColumnTransformer], Optional[Any], Optional[SelectFromModel], 
                               Optional[List[str]], Optional[Dict[str, Any]], Optional[str], 
                               Optional[pd.Series], Optional[List[str]], Optional[List[str]], 
                               Optional[pd.DataFrame], Optional[float], Optional[bool]]:
    logging.info("\n--- Chargement des Données du Modèle Sauvegardé (Logique) ---")
    
    model_pipeline_loaded: Optional[Any] = None
    preprocessor_loaded: Optional[ColumnTransformer] = None
    feature_selector_loaded: Optional[SelectFromModel] = None
    metadata: Optional[Dict[str, Any]] = None

    # Charger le préprocesseur
    preprocessor_path = FILE_PATHS['preprocessor_filename']
    if os.path.exists(preprocessor_path):
        try:
            preprocessor_loaded = joblib.load(preprocessor_path)
            logging.info(f"Préprocesseur chargé depuis {preprocessor_path}")
        except Exception as e:
            logging.error(f"Erreur lors du chargement du préprocesseur depuis {preprocessor_path}: {e}", exc_info=True)
            return None, None, None, None, None, None, None, None, None, None, None, None
    else:
        logging.error(f"Fichier préprocesseur non trouvé: {preprocessor_path}")
        return None, None, None, None, None, None, None, None, None, None, None, None

    # Charger le pipeline modèle complet
    model_path = FILE_PATHS['model_filename']
    if os.path.exists(model_path):
        try:
            model_pipeline_loaded = joblib.load(model_path)
            logging.info(f"Pipeline modèle chargé depuis {model_path}")
        except Exception as e:
            logging.error(f"Erreur lors du chargement du pipeline modèle depuis {model_path}: {e}", exc_info=True)
            return None, None, None, None, None, None, None, None, None, None, None, None
    else:
        logging.error(f"Fichier pipeline modèle non trouvé: {model_path}")
        return None, None, None, None, None, None, None, None, None, None, None, None
        
    # Charger les métadonnées
    metadata_path = FILE_PATHS['metadata_filename']
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logging.info(f"Métadonnées chargées depuis {metadata_path}")
        except Exception as e:
            logging.error(f"Erreur lors du chargement des métadonnées depuis {metadata_path}: {e}", exc_info=True)
            return None, None, None, None, None, None, None, None, None, None, None, None
    else:
        logging.error(f"Fichier métadonnées non trouvé: {metadata_path}")
        return None, None, None, None, None, None, None, None, None, None, None, None

    # Charger le sélecteur de caractéristiques (optionnel)
    fs_path = FILE_PATHS.get('feature_selector_filename') # Utiliser .get pour le cas où la clé n'existerait pas
    if fs_path and os.path.exists(fs_path):
        try:
            feature_selector_loaded = joblib.load(fs_path)
            logging.info(f"Sélecteur de caractéristiques chargé depuis {fs_path}")
        except Exception as e_fs_load:
            logging.warning(f"Impossible de charger le sélecteur de caractéristiques depuis {fs_path}: {e_fs_load}")
            feature_selector_loaded = None # Continuer même si le sélecteur n'est pas chargé
    elif fs_path:
        logging.info(f"Fichier sélecteur de caractéristiques optionnel non trouvé: {fs_path}")


    # Extraire les informations des métadonnées
    model_type_str = "Unknown"
    if model_pipeline_loaded: # S'assurer que model_pipeline_loaded n'est pas None
        base_estimator_for_type_check = None
        current_estimator_type_check = model_pipeline_loaded # Commence avec le pipeline complet
        
        # Descendre dans le pipeline pour trouver le classifieur de base
        if hasattr(current_estimator_type_check, 'named_steps'):
            if 'calibrated_classifier' in current_estimator_type_check.named_steps and \
               hasattr(current_estimator_type_check.named_steps['calibrated_classifier'], 'base_estimator'):
                current_estimator_type_check = current_estimator_type_check.named_steps['calibrated_classifier'].base_estimator
            # Après la calibration, le base_estimator peut être un autre pipeline (ex: SMOTE + classifieur)
            if hasattr(current_estimator_type_check, 'named_steps') and 'classifier' in current_estimator_type_check.named_steps:
                base_estimator_for_type_check = current_estimator_type_check.named_steps['classifier']
            elif hasattr(current_estimator_type_check, 'named_steps') and 'classifier' not in current_estimator_type_check.named_steps and \
                 len(current_estimator_type_check.steps) > 0 : # Si c'est un pipeline sans 'classifier' nommé, prendre le dernier pas
                base_estimator_for_type_check = current_estimator_type_check.steps[-1][1]

        elif CalibratedClassifierCV and isinstance(current_estimator_type_check, CalibratedClassifierCV):
             base_estimator_for_type_check = current_estimator_type_check.base_estimator
        
        suffix = ""
        if hasattr(model_pipeline_loaded, 'named_steps') and 'calibrated_classifier' in model_pipeline_loaded.named_steps:
            suffix = " (Calibré)"
        elif CalibratedClassifierCV and isinstance(model_pipeline_loaded, CalibratedClassifierCV): # Moins probable avec la structure actuelle
            suffix = " (Calibré)"

        if base_estimator_for_type_check:
            if RandomForestClassifier and isinstance(base_estimator_for_type_check, RandomForestClassifier): model_type_str = f"RandomForest{suffix}"
            elif LogisticRegression and isinstance(base_estimator_for_type_check, LogisticRegression): model_type_str = f"LogisticRegression{suffix}"
            elif LGBMClassifier is not None and isinstance(base_estimator_for_type_check, LGBMClassifier): model_type_str = f"LightGBM{suffix}"
            elif XGBClassifier is not None and isinstance(base_estimator_for_type_check, XGBClassifier): model_type_str = f"XGBoost{suffix}"
            elif suffix: model_type_str = f"Type Inconnu{suffix}"
            else: model_type_str = "Type Inconnu"
        elif suffix:  
            model_type_str = f"Type Inconnu{suffix}"
        else: # Si on n'a pas pu déterminer le type de base
            model_type_str = str(type(model_pipeline_loaded))


    X_train_dtypes_loaded_dict = metadata.get('X_train_dtypes')
    X_train_dtypes_loaded = pd.Series(X_train_dtypes_loaded_dict).astype(str) if X_train_dtypes_loaded_dict else None
    
    selected_features_initial_loaded = metadata.get('selected_features_initial')
    processed_feature_names_after_preproc_loaded = metadata.get('processed_feature_names_after_preproc')
    final_selected_feature_names_loaded = metadata.get('final_selected_feature_names') 
    optimal_threshold_youden_loaded = metadata.get('optimal_threshold_youden', 0.5) 
    feature_selection_applied_loaded = metadata.get('feature_selection_applied', False)
    
    shap_background_loaded = None
    shap_bg_path = FILE_PATHS.get('shap_background_filename')
    if shap_bg_path and os.path.exists(shap_bg_path):
        try:
            shap_background_loaded = joblib.load(shap_bg_path)
            logging.info(f"Données de fond SHAP chargées depuis {shap_bg_path}")
        except Exception as e_shap:
            logging.error(f"Erreur lors du chargement des données de fond SHAP depuis {shap_bg_path}: {e_shap}", exc_info=True)

    logging.info(f"Données du modèle chargées avec succès. Type de modèle détecté: {model_type_str}. Seuil optimal Youden J: {optimal_threshold_youden_loaded:.4f}")
    logging.info(f"Sélection de caractéristiques appliquée lors de l'entraînement: {feature_selection_applied_loaded}")
    if final_selected_feature_names_loaded:
        logging.info(f"Nombre de caractéristiques finales sélectionnées: {len(final_selected_feature_names_loaded)}")

    return (preprocessor_loaded, model_pipeline_loaded, feature_selector_loaded,
            selected_features_initial_loaded, metadata.get('feature_value_maps'), model_type_str, 
            X_train_dtypes_loaded, processed_feature_names_after_preproc_loaded, final_selected_feature_names_loaded,
            shap_background_loaded, optimal_threshold_youden_loaded, feature_selection_applied_loaded) 
    # L'erreur initiale était due à une tentative d'accès à assets[10] alors que assets pouvait être None.
    # Maintenant, on retourne None pour tous les éléments si un fichier essentiel manque.

def predict_new_patient(patient_data_dict: Dict[str, Any],
                        original_feature_columns: List[str], 
                        X_train_dtypes_original: Optional[pd.Series], 
                        preprocessor_for_shap: Any, 
                        model_pipeline: Any, 
                        optimal_threshold: float, 
                        explainer: Optional[Any] = None, 
                        feature_names_for_shap: Optional[List[str]] = None,
                        feature_selection_applied: bool = False, 
                        feature_selector_fitted: Optional[Any] = None 
                        ) -> Tuple[Optional[float], Optional[int], Optional[pd.DataFrame]]: 
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
        
        aki_probability_score = model_pipeline.predict_proba(new_patient_df)[0][1] 
        aki_prediction_binary = (aki_probability_score >= optimal_threshold).astype(int) 
        
        logging.info(f"Prédiction de probabilité IRA réussie. Score: {aki_probability_score:.4f}")
        logging.info(f"Prédiction binaire IRA (seuil {optimal_threshold:.4f}): {'Oui' if aki_prediction_binary == 1 else 'Non'}")


        shap_df_out = None
        # Vérifier si SHAP est disponible globalement et si l'explainer a été passé
        global SHAP_AVAILABLE, shap # Accéder aux variables globales pour SHAP
        if SHAP_AVAILABLE and shap and explainer is not None and feature_names_for_shap is not None:
            try:
                logging.info("Calcul des valeurs SHAP pour la prédiction...")
                new_patient_processed_df = preprocessor_for_shap.transform(new_patient_df)
                
                # S'assurer que le DataFrame a les bons noms de colonnes pour l'explainer SHAP
                # et qu'il contient uniquement les caractéristiques sur lesquelles SHAP a été entraîné.
                # `feature_names_for_shap` devrait être la liste correcte des noms après préprocessing ET sélection.
                
                # Si la sélection de caractéristiques a été appliquée, s'assurer que le DataFrame
                # passé à SHAP a les bonnes colonnes (celles sélectionnées).
                # `feature_names_for_shap` devrait déjà être cette liste.
                if list(new_patient_processed_df.columns) != feature_names_for_shap:
                    if all(fn in new_patient_processed_df.columns for fn in feature_names_for_shap):
                        new_patient_processed_df = new_patient_processed_df[feature_names_for_shap]
                    else:
                        missing_cols_for_shap = [fn for fn in feature_names_for_shap if fn not in new_patient_processed_df.columns]
                        logging.error(f"Colonnes manquantes pour SHAP après préprocessing: {missing_cols_for_shap}. "
                                      f"Colonnes disponibles: {new_patient_processed_df.columns.tolist()}. "
                                      f"Colonnes attendues pour SHAP: {feature_names_for_shap}")
                        raise ValueError("Colonnes manquantes pour SHAP après préprocessing.")
                
                # S'assurer que l'ordre des colonnes est le même que celui attendu par l'explainer
                new_patient_processed_df = new_patient_processed_df[feature_names_for_shap]


                shap_values_raw = explainer.shap_values(new_patient_processed_df) 
                
                shap_values_for_prediction: np.ndarray
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
