# Architecture du Projet Web - IA Météorologique

## Vue d'Ensemble

Ce projet est une plateforme web pour l'entraînement et la prédiction météorologique utilisant des modèles de Machine Learning. Il combine une architecture hybride avec :

- **Backend** : Django (Python) avec API RESTful
- **Frontend** : Application React (TypeScript) + Templates Django avec JavaScript vanilla
- **Base de données** : SQLite (développement)
- **Stockage** : Système de fichiers local pour datasets et modèles

## Structure du Projet

```
web_app/
├── backend/                 # Backend Flask simple (legacy)
│   ├── app.py              
│   └── requirements.txt
│
├── django_app/             # Application principale Django
│   ├── manage.py           # Script de gestion Django
│   ├── db.sqlite3          # Base de données SQLite
│   ├── requirements.txt    # Dépendances Python
│   │
│   ├── weather_prediction/ # Configuration principale Django
│   │   ├── settings.py     # Configuration du projet
│   │   ├── urls.py         # URLs principales
│   │   ├── views.py        # Vues principales
│   │   ├── auth_views.py   # Authentification
│   │   └── translations.py # Système de traductions
│   │
│   ├── ml_trainer/         # App principale de ML
│   │   ├── models.py       # Modèles de données
│   │   ├── views/          # Vues organisées par fonction
│   │   ├── urls.py         # URLs de l'API
│   │   ├── serializers.py  # Sérialisation DRF
│   │   └── ml_utils.py     # Utilitaires de ML
│   │
│   ├── templates/          # Templates Django
│   │   ├── base.html       # Template de base
│   │   └── *.html          # Templates spécifiques
│   │
│   ├── static/             # Fichiers statiques
│   │   ├── css/            # Styles
│   │   └── js/             # JavaScript
│   │
│   └── media/              # Fichiers téléchargés
│       ├── datasets/       # Datasets CSV
│       └── models/         # Modèles entraînés
│
└── frontend/               # Application React
    ├── src/
    │   ├── App.tsx         # Composant principal
    │   ├── components/     # Composants React
    │   └── contexts/       # Context API
    └── package.json        # Dépendances Node.js
```

## Architecture du Backend (Django)

### 1. Couche des Modèles (models.py)

Les modèles principaux suivent une architecture hiérarchique :

```
CustomNormalizationFunction
    └── Fonctions de normalisation personnalisées

Dataset
    └── Parent Dataset (auto-référentiel pour normalisation)
        └── Historique des transformations

ModelDefinition
    └── Template réutilisable de modèle
        └── TrainingSession (multiples entraînements)
            └── WeatherPrediction (prédictions)
```

**Caractéristiques clés :**
- Utilisation extensive de JSONField pour configurations flexibles
- Relations auto-référentielles pour le suivi des normalisations
- Références souples pour maintenir l'intégrité après suppressions
- Champs de progression pour monitoring en temps réel

### 2. Couche des Vues

Organisation modulaire dans `ml_trainer/views/` :

- **dataset_views.py** : CRUD de datasets, analyses, transformations
- **normalization_views.py** : Application de normalisations
- **model_views.py** : Gestion des définitions de modèles
- **training_views.py** : Contrôle des sessions d'entraînement
- **prediction_views.py** : Génération de prédictions
- **export_views.py** : Exportation de code et modèles

**Patterns utilisés :**
- Class-Based Views (CBV) pour opérations CRUD
- ViewSets de Django REST Framework
- Vues asynchrones pour opérations longues
- Décorateurs pour authentification et permissions

### 3. API RESTful

Structure des endpoints suivant les principes REST :

```
/api/
├── datasets/                 # Gestion des datasets
│   ├── {id}/columns/        # Opérations sur colonnes
│   ├── {id}/normalization/  # Normalisation
│   └── {id}/analysis/       # Analyse
├── models/                   # Définitions de modèles
│   └── {id}/trainings/      # Historique d'entraînements
├── training-sessions/        # Sessions d'entraînement
│   ├── {id}/train/          # Démarrer l'entraînement
│   └── {id}/results/        # Résultats
└── predictions/              # Prédictions
    └── map/                  # Visualisation géographique
```

### 4. Système de Traitement ML

**ml_utils.py & ml_utils_pytorch.py :**
- Abstraction des frameworks (Keras, PyTorch, Scikit-learn)
- Pipeline unifié d'entraînement
- Gestion des callbacks pour progression
- Sérialisation des modèles

**Flux d'entraînement :**
1. Chargement et préparation des données
2. Application des normalisations
3. Division train/val/test
4. Configuration du modèle selon framework
5. Entraînement avec callbacks de progression
6. Évaluation et sauvegarde des résultats

## Architecture du Frontend

### 1. Application React (SPA)

**Composants principaux :**
- **ModelSelection** : Sélection initiale du type de modèle
- **DatasetUpload** : Chargement de fichiers CSV avec validation
- **VariableSelection** : Sélection des prédicteurs/cibles
- **HyperparameterConfig** : Configuration spécifique par modèle
- **TrainingDashboard** : Monitoring en temps réel
- **WeatherMap** : Visualisation géographique avec Leaflet

**Gestion d'état :**
- React Context pour la langue (ES/FR/EN)
- État local dans les composants
- Passage d'état entre routes via location.state

**Intégration avec le backend :**
- Axios pour requêtes HTTP
- Polling pour mises à jour de progression
- Gestion des tokens CSRF Django
- URL de base configurable

### 2. Templates Django + JavaScript

Pour les fonctionnalités non couvertes par React :
- Gestion des datasets (datasets.html)
- Normalisation avancée (normalize.html)
- Liste des modèles (models.html)
- Dashboard principal (dashboard.html)

**JavaScript intégré :**
- dataset-analysis.js : Analyse avancée des variables
- training-progress-enhanced.js : Progression spécifique par modèle
- random-forest-config.js : Configuration spécialisée RF

## Flux de Données

### 1. Pipeline de ML

```
1. Utilisateur charge CSV → Django FileField → media/datasets/
2. Analyse des colonnes → Détection des types → Métadonnées en BD
3. Normalisation (optionnelle) → Nouveau dataset dérivé
4. Configuration du modèle → ModelDefinition en BD
5. Entraînement → TrainingSession → Callbacks de progression
6. Modèle entraîné → media/models/ → Résultats en BD
7. Prédictions → API → Visualisation sur carte
```

### 2. Authentification et Autorisation

- Django Authentication System
- Sessions basées sur cookies
- Protection CSRF
- Middleware personnalisé pour la langue
- Permissions au niveau des vues

### 3. Internationalisation

Système trilingue (FR/ES/EN) :
- Frontend : React Context + traductions dans composants
- Backend : Système de traductions Django
- Persistance dans localStorage
- Français comme langue par défaut

## Caractéristiques Techniques Remarquables

### 1. Gestion des Fichiers
- Validation des CSV lors du téléchargement
- Génération de noms uniques avec hash
- Nettoyage automatique des fichiers orphelins
- Support pour datasets volumineux

### 2. Traitement Asynchrone
- Callbacks pour progression d'entraînement
- Polling depuis le frontend (2 secondes)
- État persistant en BD
- Capacité d'arrêter les entraînements

### 3. Visualisation des Données
- Chart.js pour graphiques d'entraînement
- Leaflet pour cartes interactives
- Échelles de couleur dynamiques
- Design responsive

### 4. Optimisations
- Cache des meilleurs scores dans ModelDefinition
- Chargement paresseux des datasets volumineux
- Pagination dans les listes
- Index sur champs fréquemment consultés

## Considérations de Déploiement

### Développement
- Django Development Server (port 8000)
- React Development Server (port 3000)
- SQLite comme BD
- Debug=True pour développement

### Production (Recommandations)
- Gunicorn/uWSGI comme serveur WSGI
- Nginx comme proxy inverse
- PostgreSQL comme BD
- Séparation des fichiers static/media
- HTTPS obligatoire
- Variables d'environnement pour secrets

## Points d'Extension

1. **Nouveaux modèles ML** : Ajouter dans l'enum ModelType et ml_utils
2. **Nouvelles normalisations** : Étendre normalization_methods.py
3. **Nouveaux types de split** : Modifier data_splitter.py
4. **Nouvelles visualisations** : Ajouter composants React
5. **Nouvelles langues** : Étendre translations.py et LanguageContext

## Sécurité

- Tokens CSRF dans toutes les requêtes POST
- Validation des fichiers téléchargés
- Assainissement des entrées utilisateur
- Permissions basées sur l'utilisateur
- Non-exposition des chemins de fichiers réels
- Secrets dans variables d'environnement (production)