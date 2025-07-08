Ce repository contient un ensemble de notebooks de machine learning et de deep learning basiques, constituant un point de départ pour apprendre ces technologies.


🩺 **Projet : Breast Cancer Detection**
Détection automatique de la présence de tumeurs du sein à partir d’images médicales.


*Méthodologie*

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

*Callbacks utilisés*

    ModelCheckpoint : sauvegarde du meilleur modèle

    EarlyStopping : arrêt anticipé si pas de progrès après 5 epochs

    ReduceLROnPlateau : diminution du learning rate si la perte stagne (3 epochs)

*Évaluation*

Le modèle a été évalué sur un jeu de test en mesurant :

    Accuracy

    Precision

    Recall

    F1 Score

*Objectif*

Ce modèle est conçu pour être sauvegardé et déployé dans des environnements hospitaliers afin d’assister la détection précoce du cancer du sein.
Technologies utilisées

    Python

    TensorFlow

    Keras
    
🏃‍♂️ **Projet: Calories Burnt Prediction**

**Objectif*

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

*Méthodologie*

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

*Déploiement*

Le modèle présentant les meilleures performances est sauvegardé et prêt à être intégré dans une application pour prédire la dépense
énergétique quotidienne.

☔ **Projet : Classifying Rainy Days**

*Objectif*

Prédire si un jour est pluvieux ou non en utilisant des données météorologiques tabulaires.
Variables utilisées

    Pressure

    Temperature

    Humidity

    Cloud cover

    Sunshine

    Wind direction

    Wind speed

*Pipeline de traitement*

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

*Déploiement*

Le modèle final est prêt à être intégré dans une application permettant de prédire la pluviométrie et d’aider à la prise de décision dans l’agriculture et la météorologie.

💳 **Projet : Credit Card Fraud Detection**

*Objectif*

Détecter les fraudes par carte bancaire grâce à un modèle de classification supervisée et améliorer la sécurité des transactions.
Données

    Type : données tabulaires, stockées dans un fichier Excel

    Contenu : uniquement des variables numériques

    Problématique : fort déséquilibre des classes

        Transactions normales : 284 315

        Transactions frauduleuses : 492

*Pipeline de traitement*

    Gestion du déséquilibre

        Upsampling de la classe minoritaire pour équilibrer le dataset.

    Séparation des données

        Création d’ensembles d’entraînement et de validation.

    Entraînement des modèles

        Modèles testés :

            RandomForestClassifier

            SVC

            GaussianNB

            DecisionTreeClassifier

    Résultats

        Meilleur modèle : RandomForestClassifier

        Accuracy : 99 %

        Precision : 91 %

        Recall : 79 %

        F1 Score : 85 %

*Déploiement*

Le modèle peut être déployé dans des applications de surveillance des transactions bancaires afin de détecter et prévenir la fraude en temps réel.

✉️ **Projet : Detecting Spam Emails**

*Objectif*

Détecter automatiquement les emails indésirables afin d’améliorer l’organisation et la sécurité des boîtes de réception.
Données

    Type : données tabulaires (Excel)

    Variables :

        Message : texte des emails (anglais)

        Category : label (ham ou spam)

    Répartition :

        Ham : 4 825

        Spam : 747

*Pipeline de traitement*

    Prétraitement des labels

        Encodage :

            ham = 0

            spam = 1

    Rééquilibrage des classes

        Upsampling des messages spam pour corriger le déséquilibre.

    Nettoyage des textes

        Passage en minuscules

        Suppression des caractères non alphabétiques

        Suppression des stop words anglais

        Lemmatisation des mots

    Vectorisation

        TF-IDF pour pondérer les termes selon leur importance.

    Entraînement et évaluation

        Modèles :

            Logistic Regression

            Random Forest Classifier

        Résultat :

            Meilleur modèle : RandomForestClassifier

            Accuracy : 99 %

*Déploiement*

Ce modèle est prêt à être intégré dans une application de messagerie pour filtrer automatiquement les spams et alléger la charge de traitement des utilisateurs.

🐕 **Projet : Dog Breed Classification**

*Objectif*

Classer automatiquement des images de chiens en 70 races différentes grâce à un modèle de deep learning.
Données

    Source : fichier Excel avec les chemins des images et leurs labels

    Entraînement : 7 946 images RGB (224×224)

    Test/Validation : 700 images

    Nombre de classes : 70 races

*Pipeline de traitement*

    Prétraitement

        Normalisation des images

        Augmentation des données :

            Zoom

            Flip horizontal

    Modélisation

        ResNet101V2 pré-entraîné sur ImageNet comme extracteur de features (poids gelés)

        MLP ajouté en tête pour la classification multiclasse

    Entraînement

        Optimiseur : Adam

        Epochs : 25

        EarlyStopping avec patience = 10 epochs

*Résultats*

    Accuracy : 93%

*Déploiement*

Le modèle est prêt à être déployé dans une application pour reconnaître automatiquement la race des chiens à partir d’images.
Technologies

    Python

    TensorFlow

    Keras
