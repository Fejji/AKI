import streamlit as st
import pandas as pd
import numpy as np
import score_logic # Notre module contenant la logique de prédiction
import logging
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt # Nécessaire si on manipule les figures SHAP

# Configuration du logging pour Streamlit (optionnel mais utile)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Chargement initial du modèle et des métadonnées ---
# Cela sera fait une seule fois au démarrage de l'application Streamlit
@st.cache_resource # Utiliser st.cache_resource pour les objets lourds comme les modèles
def load_prediction_assets_streamlit(): # Renommé pour éviter conflit potentiel
    """Charge tous les actifs nécessaires à la prédiction."""
    assets = score_logic.load_model_data()
    # assets est un tuple: (preprocessor, model_pipeline, feature_selector,
    #                       selected_features_initial, feature_value_maps, model_type_str, 
    #                       X_train_dtypes, processed_feature_names_after_preproc, 
    #                       final_selected_feature_names,
    #                       shap_background_data, optimal_threshold_youden, 
    #                       feature_selection_applied_at_train_time)
    if not assets or assets[0] is None or assets[1] is None or assets[10] is None: # Vérifier préprocesseur, modèle, et seuil
        # assets[10] est optimal_threshold_youden_loaded
        logging.error("Erreur critique lors du chargement des actifs du modèle via score_logic.load_model_data()")
        return None
    return assets

prediction_assets = load_prediction_assets_streamlit()

# --- Variables globales pour les figures (si utilisées par des boutons) ---
# Ces variables seraient définies si l'entraînement se faisait dans ce script Streamlit
# Pour un modèle chargé, elles ne sont pas directement remplies ici.
# roc_fig_streamlit: Optional[plt.Figure] = None
# fi_fig_streamlit: Optional[plt.Figure] = None


if prediction_assets:
    (preprocessor_loaded, model_pipeline_loaded, feature_selector_loaded,
    selected_features_initial_loaded, feature_value_maps_loaded, loaded_model_type, 
    X_train_dtypes_loaded, processed_feature_names_after_preproc_loaded, 
    final_selected_feature_names_loaded,
    shap_background_loaded, optimal_threshold_youden_loaded, 
    feature_selection_applied_at_train_time) = prediction_assets
    
    shap_feature_names_to_use = final_selected_feature_names_loaded \
        if feature_selection_applied_at_train_time and final_selected_feature_names_loaded \
        else processed_feature_names_after_preproc_loaded

    shap_explainer = None
    if score_logic.SHAP_AVAILABLE and score_logic.shap and shap_background_loaded is not None and shap_feature_names_to_use:
        model_base_for_shap = None
        if hasattr(model_pipeline_loaded, 'named_steps'):
            if 'calibrated_classifier' in model_pipeline_loaded.named_steps and \
               hasattr(model_pipeline_loaded.named_steps['calibrated_classifier'], 'base_estimator'):
                calib_clf_step = model_pipeline_loaded.named_steps['calibrated_classifier']
                if hasattr(calib_clf_step.base_estimator, 'named_steps') and 'classifier' in calib_clf_step.base_estimator.named_steps:
                    model_base_for_shap = calib_clf_step.base_estimator.named_steps['classifier']
                else:
                    model_base_for_shap = calib_clf_step.base_estimator
            elif 'classifier' in model_pipeline_loaded.named_steps:
                 model_base_for_shap = model_pipeline_loaded.named_steps['classifier']
        elif score_logic.CalibratedClassifierCV and isinstance(model_pipeline_loaded, score_logic.CalibratedClassifierCV):
            model_base_for_shap = model_pipeline_loaded.base_estimator

        if model_base_for_shap:
            try:
                if list(shap_background_loaded.columns) != shap_feature_names_to_use:
                    if len(shap_background_loaded.columns) == len(shap_feature_names_to_use):
                        shap_background_loaded.columns = shap_feature_names_to_use
                    else:
                        st.warning(f"Discordance de colonnes pour le fond SHAP. SHAP désactivé.")
                        shap_background_loaded = None 
                
                if shap_background_loaded is not None:
                    if isinstance(model_base_for_shap, (score_logic.RandomForestClassifier, score_logic.XGBClassifier, score_logic.LGBMClassifier)):
                        shap_explainer = score_logic.shap.TreeExplainer(model_base_for_shap, shap_background_loaded)
                        logging.info("SHAP TreeExplainer initialisé pour l'application Streamlit.")
                    elif hasattr(model_base_for_shap, 'predict_proba'):
                        shap_explainer = score_logic.shap.KernelExplainer(model_base_for_shap.predict_proba, shap_background_loaded)
                        logging.info("SHAP KernelExplainer initialisé pour l'application Streamlit.")
            except Exception as e_shap_init_streamlit:
                st.warning(f"Erreur lors de l'initialisation de SHAP Explainer: {e_shap_init_streamlit}")
                shap_explainer = None
        else:
            st.warning("Impossible d'extraire le modèle de base pour SHAP Explainer.")

    # --- Interface Utilisateur Streamlit ---
    st.set_page_config(page_title="Score de Risque IRA", layout="wide")
    st.title("Calculateur de Score de Risque d'Insuffisance Rénale Aiguë (IRA)")
    st.markdown("Cet outil estime le risque d'IRA post-CEC en chirurgie cardiaque. **Il s'agit d'un outil issu de la recherche et ne remplace en aucun cas un avis médical professionnel.**")

    if not selected_features_initial_loaded:
        st.error("La liste des caractéristiques initiales n'a pas pu être chargée. L'application ne peut pas continuer.")
        st.stop()
    
    col1, col2 = st.columns([2,1]) 

    with col1:
        st.subheader("Données du Patient")
        patient_data_input: Dict[str, Any] = {}

        if selected_features_initial_loaded:
            for group_name, features_in_group in score_logic.GUI_GROUPS.items():
                actual_features_for_input = [f for f in features_in_group if f in selected_features_initial_loaded]
                if not actual_features_for_input:
                    continue

                with st.expander(group_name, expanded=True if group_name == "Patient (Antécédents)" else False):
                    for feature_name in actual_features_for_input:
                        display_name = feature_name.replace('_', ' ').capitalize()
                        
                        if feature_value_maps_loaded and feature_name in feature_value_maps_loaded and feature_value_maps_loaded[feature_name].get('labels'):
                            options_labels = feature_value_maps_loaded[feature_name]['labels']
                            options_values = feature_value_maps_loaded[feature_name]['values'] 
                            default_display = feature_value_maps_loaded[feature_name].get('default_display', options_labels[0] if options_labels else "")
                            
                            try:
                                default_index = options_labels.index(default_display)
                            except ValueError:
                                default_index = 0 

                            selected_label = st.selectbox(
                                f"{display_name}:", 
                                options=options_labels, 
                                index=default_index,
                                key=f"input_{feature_name}"
                            )
                            if selected_label:
                                label_index = options_labels.index(selected_label)
                                patient_data_input[feature_name] = options_values[label_index]
                            else:
                                 patient_data_input[feature_name] = None
                        else:
                            is_integer = False
                            if X_train_dtypes_loaded is not None and feature_name in X_train_dtypes_loaded:
                                dtype_str = str(X_train_dtypes_loaded[feature_name])
                                if 'int' in dtype_str:
                                    is_integer = True
                            
                            patient_data_input[feature_name] = st.number_input(
                                f"{display_name}:", 
                                value=None, 
                                step=1.0 if is_integer else 0.01, 
                                format="%d" if is_integer else "%.2f",
                                key=f"input_{feature_name}",
                                placeholder="Entrez valeur..."
                            )

    with col2:
        st.subheader("Résultats de la Prédiction")
        
        if st.button("Prédire le Risque IRA", type="primary", use_container_width=True):
            if not selected_features_initial_loaded:
                st.error("La liste des caractéristiques initiales n'a pas pu être chargée pour la prédiction.")
            else:
                # Vérification simple des données manquantes pour les champs numériques
                # (les dropdowns ont toujours une valeur par défaut)
                ready_for_prediction = True
                # for key, val in patient_data_input.items():
                #     if not (feature_value_maps_loaded and key in feature_value_maps_loaded): # Si ce n'est pas un dropdown
                #         if val is None: # Pour st.number_input, None signifie non rempli
                #             st.warning(f"Veuillez entrer une valeur pour: {key.replace('_', ' ').capitalize()}")
                #             ready_for_prediction = False
                #             break
                
                if ready_for_prediction:
                    # Convertir les valeurs None des number_input en NaN pour la prédiction
                    for key, val in patient_data_input.items():
                        if val is None and not (feature_value_maps_loaded and key in feature_value_maps_loaded):
                            patient_data_input[key] = np.nan

                    prob_score, binary_pred, shap_df_output = score_logic.predict_new_patient(
                        patient_data_dict=patient_data_input,
                        original_feature_columns=selected_features_initial_loaded,
                        X_train_dtypes_original=X_train_dtypes_loaded,
                        preprocessor_for_shap=preprocessor_loaded, 
                        model_pipeline=model_pipeline_loaded, 
                        optimal_threshold=optimal_threshold_youden_loaded, 
                        explainer=shap_explainer, 
                        feature_names_for_shap=shap_feature_names_to_use, 
                        feature_selection_applied=feature_selection_applied_at_train_time,
                        feature_selector_fitted=feature_selector_loaded
                    )

                    if prob_score is not None and binary_pred is not None:
                        st.metric(label="Probabilité d'IRA", value=f"{prob_score*100:.1f}%")
                        
                        pred_text = "Oui" if binary_pred == 1 else "Non"
                        st.markdown(f"**IRA Prédite (seuil {optimal_threshold_youden_loaded:.2f}): <span style='color:{'red' if binary_pred==1 else 'green'}; font-weight:bold;'>{pred_text}</span>**", unsafe_allow_html=True)

                        if prob_score * 100 > 50:
                            st.error("Risque Élevé (basé sur probabilité brute > 50%)")
                        elif prob_score * 100 > 20:
                            st.warning("Risque Modéré (basé sur probabilité brute > 20%)")
                        else:
                            st.success("Risque Faible (basé sur probabilité brute <= 20%)")

                        if shap_df_output is not None and not shap_df_output.empty:
                            st.markdown("---")
                            st.subheader("Facteurs Influents (SHAP)")
                            
                            shap_explanation_text = ""
                            top_n_shap = 5
                            
                            pos_contrib = shap_df_output[shap_df_output['SHAP Value'] > 1e-4].sort_values(by='SHAP Value', ascending=False).head(top_n_shap)
                            if not pos_contrib.empty:
                                shap_explanation_text += "**(+) Augmentant le risque:**\n"
                                for _, r_shap in pos_contrib.iterrows():
                                    shap_explanation_text += f"  - {r_shap['Feature']}: {r_shap['SHAP Value']:+.3f}\n"
                            
                            neg_contrib = shap_df_output[shap_df_output['SHAP Value'] < -1e-4].sort_values(by='SHAP Value', ascending=True).head(top_n_shap)
                            if not neg_contrib.empty:
                                shap_explanation_text += "\n**(-) Diminuant le risque:**\n"
                                for _, r_shap in neg_contrib.iterrows():
                                    shap_explanation_text += f"  - {r_shap['Feature']}: {r_shap['SHAP Value']:+.3f}\n"
                            
                            if pos_contrib.empty and neg_contrib.empty:
                                 shap_explanation_text += "(Faible influence nette des facteurs principaux pour cette prédiction)\n"
                            
                            st.text_area("Détails SHAP:", shap_explanation_text, height=200, disabled=True)

                            try:
                                if score_logic.SHAP_AVAILABLE and score_logic.shap:
                                    # Créer la figure SHAP
                                    fig_shap_summary, ax_shap_summary = plt.subplots()
                                    # Prendre les valeurs absolues pour le bar plot, mais les labels originaux
                                    shap_values_abs = shap_df_output['SHAP Value'].abs()
                                    shap_features = shap_df_output['Feature']
                                    # Trier par valeur absolue pour le graphique
                                    sorted_indices = np.argsort(shap_values_abs)[::-1]
                                    num_features_to_plot = min(10, len(shap_features)) # Afficher max 10

                                    sns.barplot(x=shap_df_output['SHAP Value'].iloc[sorted_indices[:num_features_to_plot]], 
                                                y=shap_features.iloc[sorted_indices[:num_features_to_plot]], 
                                                ax=ax_shap_summary, palette="coolwarm")
                                    ax_shap_summary.set_title("Top Facteurs SHAP pour cette Prédiction")
                                    fig_shap_summary.tight_layout()
                                    st.pyplot(fig_shap_summary)
                                    plt.close(fig_shap_summary) 
                            except Exception as e_shap_plot:
                                st.warning(f"Impossible d'afficher le graphique SHAP: {e_shap_plot}")

                        elif score_logic.SHAP_AVAILABLE and shap_explainer is None:
                             st.info("L'explainer SHAP n'a pas pu être initialisé (vérifiez les logs du serveur).")
                        elif not score_logic.SHAP_AVAILABLE:
                             st.info("La librairie SHAP n'est pas installée. Explication non disponible.")
                    else:
                        st.error("La prédiction a échoué. Veuillez vérifier les valeurs entrées et les logs du serveur.")
        
        st.markdown("---")
        st.caption(f"Modèle chargé: {loaded_model_type if loaded_model_type else 'Inconnu'}. Seuil de décision optimal (Youden J): {optimal_threshold_youden_loaded:.4f}")
        if feature_selection_applied_at_train_time:
            st.caption(f"Sélection de caractéristiques appliquée lors de l'entraînement. Caractéristiques finales utilisées: {len(final_selected_feature_names_loaded) if final_selected_feature_names_loaded else 'N/A'}")

else: # Corresponds à 'if prediction_assets:'
    st.error("L'application n'a pas pu démarrer correctement car les actifs du modèle n'ont pas été chargés.")
    st.stop()

