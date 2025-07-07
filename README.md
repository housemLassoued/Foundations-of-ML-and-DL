Ce repository contient un ensemble de notebooks de machine learning et de deep learning basiques, constituant un point de départ pour apprendre ces technologies.


🩺 Projet : Breast Cancer Detection
Détection automatique de la présence de tumeurs du sein à partir d’images médicales.


Méthodologie

Le modèle utilise le transfert learning avec InceptionV3 pré-entraîné sur ImageNet, combiné à un perceptron multicouche (MLP) servant de classificateur.
Entraînement

Le modèle a été entraîné en deux phases :

    Phase 1

        Déverrouillage complet des couches d’InceptionV3

        Optimiseur : Adam (learning rate=1e-5)

        Perte : Binary Cross Entropy

        50 epochs

    Phase 2

        Déverrouillage des 20 dernières couches uniquement

        Même configuration

        50 epochs supplémentaires

Callbacks utilisés

    ModelCheckpoint : sauvegarde du meilleur modèle

    EarlyStopping : arrêt anticipé si pas de progrès après 5 epochs

    ReduceLROnPlateau : diminution du learning rate si la perte stagne (3 epochs)

Évaluation

Le modèle a été évalué sur un jeu de test en mesurant :

    Accuracy

    Precision

    Recall

    F1 Score

Objectif

Ce modèle est conçu pour être sauvegardé et déployé dans des environnements hospitaliers afin d’assister la détection précoce du cancer du sein.
Technologies utilisées

    Python

    TensorFlow

    Keras
    
🏃‍♂️ Projet: Calories Burnt Prediction

Objectif

Prédire automatiquement la dépense calorique à partir de caractéristiques physiologiques et d’activité.
Données utilisées

    Type : données tabulaires

    Sources : fichier Excel contenant les variables :

        Gender

        Age

        Height

        Weight

        Duration

        Heart Rate

        Body Temperature

Méthodologie

    Exploration et analyse des données

        Étude des corrélations entre les variables et la cible

        Suppression des features peu contributives pour réduire la complexité

    Prétraitement

        Encodage des variables catégoriques

        Détection et traitement des valeurs aberrantes

    Séparation des données

        Ensemble d’entraînement

        Ensemble de validation

    Entraînement de modèles

        Linear Regression

        Random Forest Regressor

        XGB Regressor

    Évaluation

        Fonction de perte : Mean Absolute Error

Déploiement

Le modèle présentant les meilleures performances est sauvegardé et prêt à être intégré dans une application pour prédire la dépense énergétique quotidienne.
