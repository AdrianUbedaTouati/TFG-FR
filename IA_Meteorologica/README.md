# IA Météorologique - Plateforme de Prédiction avec Machine Learning

## 🌦️ Description du Projet

Plateforme web complète pour l'analyse et la prédiction météorologique utilisant des algorithmes de Machine Learning avancés. Le système combine une interface web intuitive avec des réseaux de neurones spécialisés pour offrir des prédictions précises de température et de conditions météorologiques.

**🏆 Meilleur Résultat**: L'architecture hiérarchique avec ensemble de vote atteint 93-94% de précision pour la classification météorologique multiclasse, représentant une amélioration significative par rapport aux approches traditionnelles.

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
- **LSTM**: Classification (99% précision pluie/neige) et régression uni/multivariable
- **Architecture Hiérarchique + Ensemble**: 93-94% précision pour classification 4 classes ⭐
- **N-BEATS**: Architecture spécialisée pour séries temporelles
- **N-HiTS**: Traitement multi-échelle hiérarchique
- **MLP**: Classification de conditions météorologiques
- **Modèles Spécialisés**: Classification binaire haute précision (93-95%)

## 🚀 Performances Clés

- **Classification multiclasse (4 classes météo)**: 93-94% avec Architecture Hiérarchique + Ensemble ⭐
- **Classification binaire (pluie/neige)**: 99.19% de précision
- **Classification par clustering**: 96.75% de précision (5 clusters météorologiques)
- **Prédiction température court terme**: 
  - Univariable: MAE = 0.176 (N-HiTS)
  - Multivariable: MAE = 0.193 (LSTM)
- **Prédiction température long terme**: 
  - Univariable: MAE = 0.285 (LSTM)
  - Multivariable: MAE = 0.275 (LSTM)
- **Support multi-horizon**: 14 jours (H=336) et 60 jours (H=1440)

## 📁 Structure du Projet

```
IA_Meteorologica/
├── web_app/
│   ├── django_app/         # Application Django principale
│   └── frontend/          # Interface React
├── Réseaux/
│   ├── Data_base_original/
│   │   ├── LSTM/          # Modèles LSTM (uni/multivariable)
│   │   ├── N-BEATS/       # Modèles N-BEATS
│   │   ├── N-HITS/        # Modèles N-HiTS
│   │   ├── MLP/           # Perceptrons multicouches
│   │   └── Clustering/    # Classification par clustering
│   └── Data_base_summaty_class/
│       └── LSTM/comparative/  # Architecture hiérarchique + ensemble
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

cd TFG-FR/IA_Meteorologica


python -m venv .venv

#Power Shell
.venv\Scripts\Activate.ps1 

#CMD
.venv\Scripts\activate.bat

#Linux/Mac
source .venv/bin/activate

# Installer les dépendances + paralelisation (carte grafique nvidia)
pip install -r requirements_cuda.txt

# Installer les dépendances 
pip install -r requirements.txt

cd web_app/django_app

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

## 🆕 Innovations Récentes

- **Architecture Hiérarchique avec Ensemble de Vote**: Décomposition du problème en sous-tâches spécialisées
- **Stratégies de Vote Multiples**: Hard, Soft, Pondéré, Confiance Pondérée, Cascade
- **Gestion Robuste des Erreurs**: Traitement des échantillons hors domaine avec classe "Other"
- **Classification par Clustering**: Identification automatique de patterns météorologiques

## 📈 Cas d'Usage

- Prédiction de température à court et long terme (uni/multivariable)
- Classification de conditions météorologiques (jusqu'à 94% de précision)
- Détection de type de précipitation (99% de précision)
- Analyse de séries temporelles météorologiques
- Classification automatique par patterns climatiques

## 🔧 Technologies Utilisées

- **Backend**: Django 4.2+, Django REST Framework
- **Frontend**: React, TypeScript, Chart.js
- **ML/DL**: PyTorch, scikit-learn, pandas
- **Base de données**: SQLite (dev), PostgreSQL (prod)
- **Déploiement**: Docker ready

---

Pour plus de détails techniques, consultez la documentation complète dans les fichiers mentionnés ci-dessus.