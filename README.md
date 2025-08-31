# IA Météorologique - Plateforme de Prédiction avec Machine Learning

## 🌦️ Description du Projet

Plateforme web complète pour l'analyse et la prédiction météorologique utilisant des algorithmes de Machine Learning avancés. Le système combine une interface web intuitive avec des réseaux de neurones spécialisés pour offrir des prédictions précises de température et de conditions météorologiques.

## 📊 Composants Principaux

### 1. Application Web
- **Backend**: Django (Python) avec API RESTful
- **Frontend**: React (TypeScript) + Templates Django
- **Base de données**: SQLite
- **Fonctionnalités principales**:
  - Gestion de datasets météorologiques
  - Normalisation et prétraitement automatique
  - Entraînement de modèles ML/DL
  - Visualisation interactive des résultats
  - Prédictions en temps réel

### 2. Réseaux de Neurones
- **LSTM**: Classification (99% précision pluie/neige) et régression multivariable
- **N-BEATS**: Architecture spécialisée pour séries temporelles
- **N-HiTS**: Traitement multi-échelle hiérarchique
- **MLP**: Classification de conditions météorologiques

## 🚀 Performances Clés

- **Classification binaire (pluie/neige)**: 99.19% de précision
- **Prédiction température court terme**: MAE = 0.176 (N-HiTS univariable)
- **Prédiction température long terme**: MAE = 0.275 (LSTM multivariable)
- **Support multi-horizon**: 14 jours (H=336) et 60 jours (H=1440)

## 📁 Structure du Projet

```
IA_Meteorologica/
├── web_app/
│   ├── django_app/         # Application Django principale
│   └── frontend/          # Interface React
├── Réseaux/
│   ├── Data_base_original/
│   │   ├── LSTM/          # Modèles LSTM
│   │   ├── N-BEATS/       # Modèles N-BEATS
│   │   ├── N-HITS/        # Modèles N-HiTS
│   │   └── MLP/           # Perceptrons multicouches
│   └── Data_base_summaty_class/
├── databases/             # Datasets météorologiques
└── documentation/         # Documentation détaillée
```

## 📚 Documentation Disponible

### Architecture et Fonctionnalités Web
- **`ARCHITECTURE_WEB.md`**: Structure technique détaillée de l'application web
- **`FONCTIONNALITES_WEB.md`**: Guide complet des fonctionnalités utilisateur

### Analyses des Réseaux de Neurones
- **`ANALYSE_RESEAUX_COMPLETE.md`**: Analyse comparative des performances de tous les modèles
- **`STRUCTURE_ARCHITECTURES_RESEAUX.md`**: Détails techniques des architectures neuronales

## 🛠️ Installation Rapide

```bash
# Cloner le repository
git clone https://github.com/AdrianUbedaTouati/TFG-FR/tree/main

# Installer les dépendances Django
cd web_app/django_app
pip install -r requirements.txt

# Migrations de base de données
python manage.py migrate

# Lancer le serveur
python manage.py runserver
```

## 🔑 Fonctionnalités Principales

### Gestion des Données
- Upload et validation de datasets CSV
- Normalisation automatique (Min-Max, Standard, personnalisée)
- Détection et traitement des valeurs manquantes
- Visualisations interactives

### Entraînement de Modèles
- Configuration flexible des hyperparamètres
- Support GPU avec mixed precision
- Validation croisée automatique
- Métriques en temps réel

### Prédictions
- Interface intuitive pour nouvelles prédictions
- Visualisation des résultats
- Export des prédictions
- API REST pour intégration

## 👥 Utilisateurs

Le système supporte plusieurs types d'utilisateurs avec permissions granulaires pour la gestion des datasets et modèles.

## 📈 Cas d'Usage

- Prédiction de température à court et long terme
- Classification de conditions météorologiques
- Détection de type de précipitation
- Analyse de séries temporelles météorologiques

## 🔧 Technologies Utilisées

- **Backend**: Django 4.2+, Django REST Framework
- **Frontend**: React, TypeScript, Chart.js
- **ML/DL**: PyTorch, scikit-learn, pandas
- **Base de données**: SQLite (dev), PostgreSQL (prod)
- **Déploiement**: Docker ready

---

Pour plus de détails techniques, consultez la documentation complète dans les fichiers mentionnés ci-dessus.