"""
Module pour le chargement et la création de pipelines de traitement de données.

Ce module fournit des utilitaires pour charger les chemins d'images et de masques
depuis une structure de répertoires, et pour créer dynamiquement des pipelines
scikit-learn à partir de configurations JSON.
"""

import os
import glob
import json
import importlib
import inspect
from typing import List, Tuple, Dict, Any, Optional
from sklearn.pipeline import Pipeline


def load_paths_data_raw(root_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Charge les chemins des fichiers d'images et de masques depuis un répertoire structuré.

    Args:
        root_dir (str): Répertoire racine contenant les sous-répertoires de classes.
                       Structure attendue : root_dir/nom_classe/images/*.png
                                          root_dir/nom_classe/masks/*.png

    Returns:
        Tuple[List[str], List[str], List[str]]: Trois listes contenant :
            - chemins_images : Liste des chemins des fichiers d'images
            - chemins_masques : Liste des chemins des masques correspondants
            - etiquettes : Liste des étiquettes de classe (en minuscules)

    Raises:
        OSError: Si root_dir n'existe pas ou n'est pas accessible.
    """
    chemins_images = []
    chemins_masques = []
    etiquettes = []

    if not os.path.exists(root_dir):
        raise OSError(f"Le répertoire racine '{root_dir}' n'existe pas")

    for etiquette in os.listdir(root_dir):
        rep_img_classe = os.path.join(root_dir, etiquette, 'images')
        rep_masque_classe = os.path.join(root_dir, etiquette, 'masks')

        if os.path.isdir(rep_img_classe):
            for chemin_img in glob.glob(os.path.join(rep_img_classe, '*.png')):
                nom_fichier = os.path.basename(chemin_img)
                chemin_masque = os.path.join(rep_masque_classe, nom_fichier)

                if os.path.exists(chemin_masque):
                    chemins_images.append(chemin_img)
                    chemins_masques.append(chemin_masque)
                    etiquettes.append(etiquette.lower())

    return chemins_images, chemins_masques, etiquettes


def load_pipeline_by_name(nom_pipeline: str, masques: Optional[List[str]] = None) -> Pipeline:
    """
    Charge un pipeline par son nom depuis le registre.

    Args:
        nom_pipeline (str): Nom du pipeline à charger
        masques (Optional[List[str]]): Liste optionnelle de chemins de masques à injecter

    Returns:
        Pipeline: Objet Pipeline scikit-learn configuré

    Raises:
        ValueError: Si nom_pipeline n'est pas trouvé dans le registre
        FileNotFoundError: Si les fichiers de configuration ne sont pas trouvés
    """
    chemin_config = "../../src/features/Pipelines/Configs_Pipelines/pipeline_config.json"
    config_registre = load_pipeline_config(chemin_config)

    # Filtrer les entrées de métadonnées
    pipelines_disponibles = {
        k: v for k, v in config_registre.items()
        if isinstance(v, dict) and "file" in v
    }

    if nom_pipeline not in pipelines_disponibles:
        disponibles = list(pipelines_disponibles.keys())
        raise ValueError(
            f"Pipeline '{nom_pipeline}' non trouvé. Disponibles : {disponibles}"
        )

    fichier_pipeline = pipelines_disponibles[nom_pipeline]["file"]
    config_pipeline = load_pipeline_config(fichier_pipeline)

    print(f"Chargement du pipeline : {config_pipeline['name']}")
    print(f"Description : {config_pipeline['description']}")

    return create_pipeline_from_config(config_pipeline, masques)


def get_transformer_class(nom_classe: str) -> type:
    """
    Récupère une classe de transformateur depuis différents contextes.

    Cherche d'abord dans le scope global du module appelant,
    puis dans les modules de transformateurs.

    Args:
        nom_classe (str): Nom de la classe de transformateur

    Returns:
        type: La classe de transformateur

    Raises:
        NameError: Si nom_classe n'est pas trouvé
    """
    # 1. Chercher dans le scope global du module appelant
    frame = inspect.currentframe()
    try:
        # Remonter la pile d'appels pour trouver le bon contexte
        caller_frame = frame.f_back
        while caller_frame:
            caller_globals = caller_frame.f_globals
            if nom_classe in caller_globals:
                return caller_globals[nom_classe]
            caller_frame = caller_frame.f_back
    finally:
        del frame

    # 2. Chercher dans les modules de transformateurs
    modules_transformateurs = [
        'src.features.Pipelines.Transformateurs.image_loaders',
        'src.features.Pipelines.Transformateurs.image_preprocessing',
        'src.features.Pipelines.Transformateurs.image_features',
        'src.features.Pipelines.Transformateurs.image_augmentation',
        'src.features.Pipelines.Transformateurs.utilities',
        'src.features.Pipelines.Transformateurs.tensorflow_transformers'
    ]

    for module_name in modules_transformateurs:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, nom_classe):
                return getattr(module, nom_classe)
        except ImportError:
            continue

    # 3. Si rien n'est trouvé, lever une exception
    raise NameError(f"Classe '{nom_classe}' non trouvée dans le scope global ou les modules")


def load_pipeline_config(fichier_pipeline: str) -> Dict[str, Any]:
    """
    Charge la configuration d'un pipeline depuis un fichier JSON.

    Args:
        fichier_pipeline (str): Chemin vers le fichier de configuration JSON

    Returns:
        Dict[str, Any]: Dictionnaire de configuration du pipeline

    Raises:
        FileNotFoundError: Si le fichier de configuration n'existe pas
        json.JSONDecodeError: Si le fichier contient du JSON invalide
    """
    try:
        with open(fichier_pipeline, 'r', encoding='utf-8') as fichier:
            return json.load(fichier)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Fichier de config pipeline '{fichier_pipeline}' non trouvé"
        ) from exc
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"JSON invalide dans le fichier de config '{fichier_pipeline}'",
            exc.doc,
            exc.pos
        ) from exc


def create_pipeline_from_config(config: Dict[str, Any],
                               masques: Optional[List[str]] = None) -> Pipeline:
    """
    Crée un pipeline scikit-learn à partir d'une configuration JSON.
    Gère les pipelines simples et composites.

    Args:
        config (Dict[str, Any]): Dictionnaire de configuration du pipeline
        masques (Optional[List[str]]): Chemins de masques optionnels à injecter

    Returns:
        Pipeline: Objet Pipeline scikit-learn configuré

    Raises:
        ValueError: Si le type de pipeline n'est pas supporté
    """
    type_pipeline = config.get("type", "simple")

    if type_pipeline == "composite":
        return create_composite_pipeline(config, masques)
    if type_pipeline == "simple":
        return create_simple_pipeline(config, masques)
    raise ValueError(f"Type de pipeline non supporté : {type_pipeline}")


def create_simple_pipeline(config: Dict[str, Any],
                          masques: Optional[List[str]] = None) -> Pipeline:
    """
    Crée un pipeline simple à partir des définitions d'étapes.

    Args:
        config (Dict[str, Any]): Configuration du pipeline contenant les étapes
        masques (Optional[List[str]]): Chemins de masques optionnels à injecter

    Returns:
        Pipeline: Objet Pipeline scikit-learn configuré

    Raises:
        KeyError: Si les clés de configuration requises sont manquantes
        NameError: Si les classes de transformateur ne sont pas trouvées
    """
    etapes = []

    for etape in config["steps"]:
        classe_transformateur = get_transformer_class(etape["class"])
        parametres = etape["params"].copy()  # Copie pour éviter de modifier l'original

        # Injecter les chemins de masques si nécessaire
        if ("mask_paths" in parametres and
            parametres["mask_paths"] == "masks" and
            masques is not None):
            parametres["mask_paths"] = masques

        etapes.append((etape["name"], classe_transformateur(**parametres)))

    return Pipeline(etapes, verbose=True)


def create_composite_pipeline(config: Dict[str, Any],
                             masques: Optional[List[str]] = None) -> Pipeline:
    """
    Crée un pipeline composite en combinant d'autres pipelines.

    Args:
        config (Dict[str, Any]): Configuration du pipeline composite
        masques (Optional[List[str]]): Chemins de masques optionnels à injecter

    Returns:
        Pipeline: Objet Pipeline scikit-learn configuré

    Raises:
        KeyError: Si les clés de configuration requises sont manquantes
        FileNotFoundError: Si les fichiers de sous-pipelines ne sont pas trouvés
        NameError: Si les classes de transformateur ne sont pas trouvées
    """
    toutes_etapes = []

    for etape in config["steps"]:
        if etape.get("type") == "pipeline":
            # Charger le sous-pipeline
            sous_config = load_pipeline_config(etape["pipeline_file"])

            # Filtrer les étapes si nécessaire
            sous_etapes = sous_config["steps"]
            if "include_steps" in etape:
                sous_etapes = [s for s in sous_etapes if s["name"] in etape["include_steps"]]
            elif "exclude_steps" in etape:
                sous_etapes = [s for s in sous_etapes if s["name"] not in etape["exclude_steps"]]

            # Ajouter les étapes du sous-pipeline
            for sous_etape in sous_etapes:
                classe_transformateur = get_transformer_class(sous_etape["class"])
                parametres = sous_etape["params"].copy()  # Copie pour éviter de modifier l'original

                # Injecter les chemins de masques si nécessaire
                if ("mask_paths" in parametres and
                    parametres["mask_paths"] == "masks" and
                    masques is not None):
                    parametres["mask_paths"] = masques

                nom_etape = f"{etape['name']}_{sous_etape['name']}"
                toutes_etapes.append((nom_etape, classe_transformateur(**parametres)))

        else:
            # Étape normale
            classe_transformateur = get_transformer_class(etape["class"])
            parametres = etape["params"].copy()  # Copie pour éviter de modifier l'original

            # Injecter les chemins de masques si nécessaire
            if ("mask_paths" in parametres and
                parametres["mask_paths"] == "masks" and
                masques is not None):
                parametres["mask_paths"] = masques

            toutes_etapes.append((etape["name"], classe_transformateur(**parametres)))

    return Pipeline(toutes_etapes, verbose=True)
