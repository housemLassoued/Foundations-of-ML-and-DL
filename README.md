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

Le modèle présentant les meilleures performances est sauvegardé et prêt à être intégré dans une application pour prédire la dépense

☔ Classifying Rainy Days

Objectif

Prédire si un jour est pluvieux ou non en utilisant des données météorologiques tabulaires.
Variables utilisées

    Pressure

    Temperature

    Humidity

    Cloud cover

    Sunshine

    Wind direction

    Wind speed

Pipeline de traitement

    Analyse des données

        Étude des corrélations avec la variable cible (rainfall).

        Suppression des variables à faible impact.

    Prétraitement

        Encodage des variables catégoriques.

        Traitement des valeurs aberrantes et des valeurs manquantes.

        Normalisation des features (moyenne = 0, écart-type = 1).

    Gestion du déséquilibre des classes

        Upsampling de la classe minoritaire (no rain) pour équilibrer les classes.

    Entraînement et validation

        Séparation des données en training set et validation set.

        Modèles testés :

            RandomForestClassifier

            SVC

            XGBClassifier

            DecisionTreeClassifier

    Résultats du meilleur modèle

        Modèle choisi : RandomForestClassifier

        Accuracy : 87%

        Precision : 91%

        Recall : 82%

        F1 Score : 86%

Déploiement

Le modèle final est prêt à être intégré dans une application permettant de prédire la pluviométrie et d’aider à la prise de décision dans l’agriculture et la météorologie.

énergétique quotidienne.
