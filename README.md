# Création d'un contenu README basé sur les informations extraites


# Fracture Detection with MobileNetV2

Ce projet utilise l'apprentissage profond pour classifier des images médicales en deux catégories : **fracturée** et **non fracturée**. Il s'appuie sur un modèle pré-entraîné (MobileNetV2) adapté pour la classification binaire d'images.

##  Objectif

Détecter automatiquement les fractures osseuses à partir d'images médicales, afin d'assister les professionnels de santé dans leur diagnostic.

##  Structure des données

Les images sont classées dans deux dossiers :
- `1_Fractured/` : images avec fracture
- `0_Non_fractured/` : images sans fracture

##  Technologies

- Python
- TensorFlow / Keras
- scikit-learn
- matplotlib / seaborn
- NumPy
- PIL (Pillow)

##  Pipeline du projet

1. **Chargement et affichage des données**
   - Chargement aléatoire d'échantillons pour vérification visuelle

2. **Prétraitement et augmentation des données**
   - Redimensionnement, rotation, zoom, flip horizontal
   - Séparation entraînement / validation (split 70% / 30%)

3. **Modèle MobileNetV2**
   - Chargé sans sa couche de sortie
   - Ajout de : `GlobalAveragePooling2D`, `Dropout`, `Dense`
   - Fonction de perte personnalisée : **Focal Loss**

4. **Entraînement**
   - Optimiseur : Adam
   - Callbacks : EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

5. **Évaluation**
   - Matrice de confusion
   - Classification report (Precision, Recall, F1)
   - Courbe ROC & AUC

##  Résultats

Le modèle est évalué à l'aide d'indicateurs classiques de classification. Des visualisations (matrices de confusion, courbes ROC) permettent de juger de la performance.

##  Remarques

- Ce projet peut être adapté à d'autres types d'images médicales avec peu de modifications.
- La `focal loss` est particulièrement utile pour les jeux de données déséquilibré
