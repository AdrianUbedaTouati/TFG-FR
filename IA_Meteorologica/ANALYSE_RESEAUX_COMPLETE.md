# Analyse Complète des Réseaux de Neurones - Projet Météorologique

## Résumé Exécutif

Ce document présente une analyse exhaustive des différentes architectures de réseaux de neurones implémentées dans le projet météorologique. On analyse 5 types principaux de réseaux : LSTM (classification et régression), N-BEATS, N-HiTS, MLP et réseaux de classification de couverture nuageuse.

## 1. RÉSEAUX LSTM

### 1.1 LSTM pour Classification

#### 1.1.1 Réseau Ventana (Classification du Résumé Météorologique - 5 classes)
- **Configuration** : H=336, L=24, C=5
- **Variable cible** : Summary (5 classes : Partly Cloudy, Mostly Cloudy, Overcast, Clear, Foggy)
- **Précision** : 26,93%
- **Caractéristiques** : Faible précision due à la complexité de distinguer entre 5 types de conditions climatiques

#### 1.1.2 Réseau Rain (Classification Binaire Pluie/Neige)
- **Variable cible** : Type de précipitation (rain/snow)
- **Précision** : 99,19%
- **Score F1** : 0,9662
- **Caractéristiques** : Excellente performance en classification binaire, dataset déséquilibré (rain >> snow)

#### 1.1.3 Réseau Summary (Classification du Résumé - 3 classes)
- **Variable cible** : Summary (3 classes : Partly Cloudy, Mostly Cloudy, Overcast)
- **Précision** : 56,77%
- **Score F1** : 0,5566
- **Caractéristiques** : Performance modérée, version simplifiée du réseau Ventana

### 1.2 LSTM Multivariable (Régression)

#### 1.2.1 Court/Neutral (Prédiction à court terme)
- **Configuration** : H=336, L=24 (14 jours horizon, 1 jour lookback)
- **Variable cible** : Température (°C) normalisée
- **Métriques normalisées** : RMSE=0,250, MAE=0,193

#### 1.2.2 Long/Neutral (Prédiction à long terme)
- **Configuration** : H=1440, L=120 (60 jours horizon, 5 jours lookback)
- **Variable cible** : Température (°C) normalisée
- **Métriques normalisées** : RMSE=0,344, MAE=0,275

#### 1.2.3 Long/Ventanas
- **Configuration** : H=1440, L=120
- **Variable cible** : Température (°C) normalisée
- **Métriques normalisées** : RMSE=0,349, MAE=0,284
- **Caractéristiques** : Performance inférieure au neutral, suggère que la stratégie de fenêtres n'est pas optimale

## 2. RÉSEAUX N-BEATS

### 2.1 N-BEATS Univariable

#### 2.1.1 Court (H=336, L=24)
- **Neutral** : RMSE=0,260, MAE=0,188
- **Grid** : RMSE=0,258, MAE=0,186 (meilleur)
- **Random** : RMSE=0,258, MAE=0,191
- **Meilleurs hyperparamètres** : Hidden=256, Depth=4, Blocks=8

#### 2.1.2 Long (H=1440, L=120)
- **Neutral** : RMSE=0,407, MAE=0,308
- **Random** : RMSE=0,398, MAE=0,308 (meilleur)
- **Meilleurs hyperparamètres** : Hidden=256, Depth=4, Blocks=4

### 2.2 N-BEATS Multivariable

#### 2.2.1 Court (H=336, L=24)
- **Neutral** : RMSE=0,395, MAE=0,319
- **Random** : RMSE=0,297, MAE=0,230 (meilleur)
- **Caractéristiques** : 19 features incluant conditions météorologiques

#### 2.2.2 Long (H=1440, L=120)
- **Random** : RMSE=0,425, MAE=0,338 

**Observation clé** : Pour N-BEATS, les modèles univariables surpassent significativement les multivariables en prédictions à court terme.

## 3. RÉSEAUX N-HiTS

### 3.1 Comparaison Court (H=336, L=24)

| Type | Configuration | Val Loss | MAE | RMSE |
|------|---------------|----------|-----|------|
| Univariable/Neutral | D=1 | 0,0268 | 0,1764 | 0,2408 |
| Univariable/Random | D=1 | 0,0275 | 0,1759 | 0,2399 |
| Multivariable/Random | D=21 | 0,0268 | 0,2284 | 0,2863 |

### 3.2 Comparaison Long (H=1440, L=120)

| Type | Configuration | Val Loss | MAE | RMSE |
|------|---------------|----------|-----|------|
| Univariable/Neutral | D=1 | 0,0657 | 0,3345 | 0,4169 |
| Univariable/Random | D=1 | 0,0632 | 0,3296 | 0,4115 |
| Multivariable/Neutral | D=21 | 0,0664 | 0,3195 | 0,4030 |

**Observation clé** : À la différence de N-BEATS, dans N-HiTS les modèles multivariables montrent un avantage pour les prédictions à long terme.

## 4. RÉSEAUX MLP

### 4.1 MLP pour Classification de Summary (3 classes)
- **Classes** : Cloudy, Clear, Foggy
- **Distribution** : Très déséquilibrée (Cloudy >> Clear > Foggy)
- **Architecture** : Réseau neuronal feedforward simple
- **Note** : Aucune métrique de précision trouvée dans les fichiers disponibles

### 4.2 MLP pour Cloud Cover (4 classes)
- **Classes** : clear, cloudy, overcast, partly cloudy
- **Précision (sans CV)** : 61,29%
- **Score F1** : 0,580
- **Précision (avec CV)** : 62,12%
- **Score F1 CV** : 0,584
- **Caractéristiques** : La validation croisée améliore légèrement la performance

## 5. COMPARAISONS ET SIMILITUDES

### 5.1 Patterns Architecturaux

1. **LSTM vs N-BEATS/N-HiTS** : 
   - LSTM utilise une architecture récurrente traditionnelle
   - N-BEATS/N-HiTS utilisent des blocs spécialisés pour séries temporelles

2. **Configurations H/L standard** :
   - Court : H=336, L=24 (14 jours prédiction, 1 jour historique)
   - Long : H=1440, L=120 (60 jours prédiction, 5 jours historique)

### 5.2 Performance Comparative

#### Pour Classification :
1. **Binaire (rain/snow)** : LSTM 99,19% - Excellent
2. **3 classes (summary)** : LSTM 56,77% - Modéré
3. **4 classes (cloud cover)** : MLP 62,12% - Modéré
4. **5 classes (summary complet)** : LSTM 26,93% - Faible

#### Pour Régression (Température) :

**CLASSEMENT COURT TERME (H=336, L=24)**

**Univariable :**
1. N-HiTS : MAE=0,176, RMSE=0,240
2. N-BEATS : MAE=0,186, RMSE=0,258
3. (LSTM n'a pas de version univariable)

**Multivariable :**
1. LSTM : MAE=0,193, RMSE=0,250 ⭐ MEILLEUR
2. N-HiTS : MAE=0,228, RMSE=0,286
3. N-BEATS : MAE=0,230, RMSE=0,297

**CLASSEMENT LONG TERME (H=1440, L=120)**

**Univariable :**
1. N-BEATS : MAE=0,308, RMSE=0,398 ⭐ MEILLEUR
2. N-HiTS : MAE=0,330, RMSE=0,412

**Multivariable :**
1. LSTM : MAE=0,275, RMSE=0,344 ⭐ MEILLEUR
2. N-HiTS : MAE=0,320, RMSE=0,403
3. N-BEATS : MAE=0,338, RMSE=0,425

### 5.3 Recommandations Actualisées

1. **Pour classification binaire** : LSTM est la meilleure option (99,19% précision)
2. **Pour classification multiclasse** : MLP avec CV ou réduire le nombre de classes
3. **Pour prédiction de température à court terme** :
   - Univariable : N-HiTS (MAE=0,176)
   - Multivariable : LSTM (MAE=0,193)
4. **Pour prédiction de température à long terme** :
   - Univariable : N-BEATS (MAE=0,308)
   - Multivariable : LSTM (MAE=0,275)
5. **Conclusion importante** : LSTM multivariable est constamment le meilleur modèle pour régression quand on utilise plusieurs features

## 6. MÉTRIQUES UTILISÉES ET LEUR JUSTIFICATION

### 6.1 Pour Problèmes de Classification

#### Précision (Accuracy)
- **Ce qu'elle mesure** : Pourcentage de prédictions correctes sur le total
- **Quand l'utiliser** : Quand les classes sont équilibrées
- **Limitation** : Peut être trompeuse avec des classes déséquilibrées

#### Score F1
- **Ce qu'il mesure** : Moyenne harmonique entre précision et rappel
- **Pourquoi l'utiliser** : Équilibre faux positifs et faux négatifs
- **Avantage** : Meilleure métrique pour datasets déséquilibrés (ex : rain/snow)
- **Plage** : 0-1, où 1 est parfait

### 6.2 Pour Problèmes de Régression

#### MAE (Mean Absolute Error)
- **Ce qu'elle mesure** : Moyenne des erreurs absolues
- **Pourquoi l'utiliser** : Facile à interpréter, robuste aux valeurs aberrantes
- **Interprétation** : Erreur moyenne dans les mêmes unités que la variable cible
- **Exemple** : MAE=0,193 signifie erreur moyenne de 19,3% de la plage normalisée

#### RMSE (Root Mean Square Error)
- **Ce qu'elle mesure** : Racine de la moyenne des erreurs carrées
- **Pourquoi l'utiliser** : Pénalise plus les grandes erreurs
- **Avantage** : Utile quand les grandes erreurs sont particulièrement problématiques
- **Comparaison** : Toujours RMSE ≥ MAE ; si RMSE >> MAE, il y a de grandes erreurs occasionnelles

### 6.3 Métriques Normalisées vs Non Normalisées
- **Normalisées (0-1)** : Permettent de comparer entre différents modèles et datasets
- **Non normalisées (°C)** : Plus interprétables pour l'utilisateur final
- **Conversion** : On peut estimer l'erreur réelle en multipliant par la plage de la variable

## 7. TECHNIQUES POUR AMÉLIORER LA PRÉCISION

### 7.1 Normalisation des Données
- **Pourquoi** : Les algorithmes de deep learning convergent mieux quand les données sont dans des plages similaires (0-1)
- **Comment** : Min-Max scaling ou StandardScaler pour amener toutes les variables à la même plage
- **Impact** : Essentiel pour l'entraînement stable et la comparaison entre métriques

### 7.2 Ingénierie des Caractéristiques Temporelles
- **Codage cyclique** : Utiliser sin/cos pour heure du jour et jour de l'année
- **Pourquoi** : Capture la nature cyclique du temps (ex : 23:59 est proche de 00:00)
- **Exemple** : hour_sin = sin(2π * hour/24), hour_cos = cos(2π * hour/24)

### 7.3 Recherche d'Hyperparamètres
- **Grid Search** : Test systématique de combinaisons prédéfinies
- **Random Search** : Exploration aléatoire de l'espace des hyperparamètres
- **Pourquoi** : Trouve la configuration optimale de learning rate, couches cachées, blocs, etc.
- **Résultat** : Améliorations de 5-15% en précision typiquement

### 7.4 Architectures Spécialisées
- **N-BEATS/N-HiTS** : Conçues spécifiquement pour séries temporelles
- **Blocs résiduels** : Permettent d'entraîner des réseaux plus profonds sans dégradation
- **Attention temporelle** : Permet au modèle de se concentrer sur les moments pertinents du passé

### 7.5 Stratégies d'Entraînement
- **Early Stopping** : Arrête l'entraînement quand la validation ne s'améliore pas
- **Learning Rate Scheduling** : Réduit le taux d'apprentissage graduellement
- **Régularisation** : Weight decay (L2) prévient le surajustement

### 7.6 Sélection de Features
- **Univariable vs Multivariable** : 
  - Court terme : Moins de features peuvent réduire le bruit
  - Long terme : Plus de features capturent des patterns complexes
- **Feature Importance** : Identifie quelles variables contribuent le plus aux prédictions

### 7.7 Gestion des Données Déséquilibrées
- **Problème** : En classification rain/snow il y a beaucoup plus de données de pluie
- **Solution** : Class weighting ou échantillonnage stratifié
- **Résultat** : Meilleure performance sur les classes minoritaires

### 7.8 Validation Croisée (Cross-Validation)
- **K-fold CV** : Divise les données en K parties, entraîne K modèles
- **Pourquoi** : Estimation plus robuste de la performance réelle
- **Exemple** : MLP cloud cover amélioré de 61,29% à 62,12% avec CV

## 8. CONCLUSIONS FINALES

1. **LSTM multivariable domine en régression** : Contrairement aux attentes, LSTM surpasse N-BEATS et N-HiTS en configurations multivariables
2. **Pour univariable** : N-HiTS est meilleur à court terme, N-BEATS à long terme
3. **La complexité importe** : La performance décroît avec plus de classes (99%→57%→27%)
4. **Horizon temporel** : Les erreurs augmentent ~40% en passant du court au long terme
5. **La stratégie de "ventanas" dans LSTM n'a pas montré d'améliorations**
6. **LSTM est versatile** : Excellent en classification binaire (99%) et leader en régression multivariable
7. **Pas d'architecture universelle** : Le choix dépend du type de problème et configuration