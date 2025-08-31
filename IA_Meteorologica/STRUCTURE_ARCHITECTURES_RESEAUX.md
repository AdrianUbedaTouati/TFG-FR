# Structure et Architecture des Réseaux de Neurones

Ce document détaille l'architecture interne de chaque type de réseau de neurones implémenté dans le projet météorologique.

## 1. RÉSEAUX LSTM (Long Short-Term Memory)

### Architecture de Base
```
Entrée → Couches LSTM → Tête de Sortie
```

#### Composants Principaux :
- **Couches LSTM** : 2 couches bidirectionnelles (par défaut)
- **Unités cachées** : 256 par couche
- **Dropout** : 0,2 entre couches LSTM
- **Activation** : Tanh (interne LSTM) + GELU (dans tête MLP optionnelle)

### Variantes :

#### 1.1 LSTM pour Régression (`LSTMForecaster`)
- **Entrée** : Séquence temporelle de caractéristiques météorologiques
- **Sortie** : Prédictions de température pour H pas futurs
- **Fonction de perte** : Weighted Huber Loss
- **Tête optionnelle** : MLP avec GELU pour projection finale

#### 1.2 LSTM pour Classification (`LSTMClassifier`)
- **Différences avec régression** :
  - Tête de sortie : Linear → Softmax
  - Fonction de perte : Weighted CrossEntropy
  - Sortie : Probabilités pour C classes

## 2. N-BEATS (Neural Basis Expansion Analysis)

### Architecture de Base
```
Entrée → Stack de Blocs N-BEATS → Agrégation → Prévision
```

#### Composants de chaque Bloc :
1. **MLP d'entrée** : 
   - Profondeur configurable (défaut : 2 couches)
   - Largeur : 256 unités
   - Activation : ReLU

2. **Projections duales** :
   - Backcast : Pour reconstruire le passé
   - Forecast : Pour prédire le futur

3. **Connexions résiduelles** : Entre blocs consécutifs

#### Configuration typique :
- **Blocs** : 6 (court) ou 8 (long)
- **Largeur cachée** : 256
- **Profondeur par bloc** : 2-4 couches

### Différences selon configuration :
- **Univariable** : Traite uniquement la température
- **Multivariable** : Traite 19+ caractéristiques météorologiques

## 3. N-HiTS (Neural Hierarchical Interpolation for Time Series)

### Architecture de Base
```
Entrée → Traitement Multi-Échelle → Interpolation Hiérarchique → Prévision
```

#### Composants Principaux :

1. **Pooling Multi-échelle** :
   - Échelles : [1, 2, 4, 8] (différentes résolutions temporelles)
   - Type : Average pooling

2. **Blocs N-HiTS** (pour chaque échelle) :
   - MLP avec GELU et Dropout(0,1)
   - Tête linéaire pour prédictions grossières
   - Interpolation à la résolution cible

3. **Paramètres clés** :
   - Blocs par échelle : 2
   - Largeur cachée : 256
   - Profondeur (D) : 1 (univariable) ou 21 (multivariable)

### Avantage architectural :
Traite le signal à plusieurs résolutions simultanément, capturant des patterns de court et long terme.

## 4. MLP (Multi-Layer Perceptron)

### 4.1 MLP pour Classification Météorologique

#### Architecture Profonde :
```
Entrée → [Linear(1536) → LayerNorm → GELU → Dropout] × 6 → Sortie(3)
```

#### Caractéristiques spéciales :
- **Connexions résiduelles** : Toutes les 2 couches
- **Normalisation** : LayerNorm après chaque couche
- **Régularisation** : Dropout(0,3) agressif

### 4.2 MLP pour Cloud Cover

#### Architecture Flexible :
```
Entrée → Linear(512) → ReLU → BatchNorm → Linear(256) → ReLU → BatchNorm → Linear(128) → Sortie(4)
```

#### Différences avec MLP météorologique :
- Moins de couches (3 vs 6)
- BatchNorm au lieu de LayerNorm
- Sans connexions résiduelles
- Architecture plus simple mais efficace

## 5. OPTIMISATION ET ENTRAÎNEMENT

### Configurations Communes :

1. **Optimiseur** : AdamW
   - Learning rate : 2e-3 à 3e-4
   - Weight decay : 1e-5 à 1e-6

2. **Techniques GPU** :
   - Mixed Precision (AMP)
   - TF32 pour Ampere+
   - Transferts non-bloquants

3. **Régularisation** :
   - Early stopping (patience : 10-50 époques)
   - Gradient clipping (norme : 1,0)
   - Learning rate scheduling

4. **Gestion des données** :
   - Fenêtres glissantes pour séries temporelles
   - Normalisation Min-Max ou Standard
   - Pondération des classes pour déséquilibre

## 6. DIFFÉRENCES CLÉS ENTRE ARCHITECTURES

### LSTM vs N-BEATS/N-HiTS :
- LSTM : Traitement séquentiel avec mémoire
- N-BEATS/N-HiTS : Traitement parallèle avec blocs spécialisés

### Univariable vs Multivariable :
- **Univariable** : Architecture plus simple, moins de paramètres
- **Multivariable** : Plus grande profondeur (D=21 vs D=1 dans N-HiTS)

### Court vs Long :
- Principalement différent en configuration, pas en architecture
- Long peut utiliser plus de blocs ou plus de profondeur

## 7. SÉLECTION D'ARCHITECTURE

### Critères de décision :

1. **Pour classification binaire** : LSTM pour sa capacité séquentielle
2. **Pour séries temporelles univariables** : N-HiTS pour son traitement multi-échelle
3. **Pour multiples features** : LSTM multivariable pour sa flexibilité
4. **Pour classification multiclasse simple** : MLP avec architecture profonde

### Compromis :
- **Complexité vs Performance** : N-HiTS plus complexe mais meilleur pour patterns multi-échelle
- **Vitesse vs Précision** : MLP plus rapide, LSTM plus précis pour séquences
- **Interprétabilité** : N-BEATS offre décomposition backcast/forecast