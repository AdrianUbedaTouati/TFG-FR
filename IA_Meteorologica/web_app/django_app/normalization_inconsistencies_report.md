# Normalization Implementation Inconsistencies Report

## Overview
The normalization implementation in the Django ML trainer application shows several inconsistencies and unused code patterns across multiple files.

## Key Findings

### 1. **Unused `Normalizador` Class in `normalization_views.py`**
- **File**: `/ml_trainer/views/normalization_views.py`
- **Issue**: The `Normalizador` class is imported (line 13) but never used
- **Current Implementation**: 
  - Lines 88 and 93 create `NumNorm` and `TextNorm` instances directly with string parameters
  - The `Normalizador` class in `normalization_methods.py` (lines 119-138) is designed to handle normalization but is ignored

### 2. **Multiple Normalization Implementations**
The codebase has **THREE different normalization systems**:

#### A. Django Model Choices (`models.py`)
```python
class NormalizationMethod(models.TextChoices):
    MIN_MAX = 'min_max', 'Min-Max'
    STANDARD = 'standard', 'Estandarización'
    ROBUST = 'robust', 'Robust Scaler'
    NONE = 'none', 'Sin normalización'
```

#### B. Enum-based System (`normalization_methods.py`)
```python
class NumNorm(Enum):
    MIN_MAX = auto()
    Z_SCORE = auto()
    LSTM_TCN = auto()
    CNN = auto()
    TRANSFORMER = auto()
    TREE = auto()

class TextNorm(Enum):
    LOWER = auto()
    STRIP = auto()
    ONE_HOT = auto()
```

#### C. String-based System (`ml_utils.py`)
```python
def get_scaler(method):
    if method == 'min_max':
        return MinMaxScaler()
    elif method == 'standard':
        return StandardScaler()
    elif method == 'robust':
        return RobustScaler()
```

### 3. **Incorrect Usage in `normalization_views.py`**
- **Lines 88-93**: Passing string methods to enum constructors
  ```python
  normalizer = NumNorm(method)  # method is a string, not an enum value
  normalizer = TextNorm(method)
  ```
- This will fail because `NumNorm` and `TextNorm` are Enums, not classes with `__init__` methods

### 4. **Proper Usage in `views.py`**
The main `views.py` file uses the normalization correctly:
- Lines 2044-2047: Uses `NumNorm[method]` to get enum value
- Lines 2057-2060: Uses `TextNorm[method]` to get enum value
- Lines 2045, 2058: Properly instantiates `Normalizador` class

### 5. **Inconsistent Method Names**
Different naming conventions across the codebase:
- Django models: `'min_max'`, `'standard'`, `'robust'`
- Enum system: `MIN_MAX`, `Z_SCORE`, `LSTM_TCN`, etc.
- Frontend expects: `'onehot'`, `'label'` (line 92 in normalization_views.py)

### 6. **Missing Imports and Error Handling**
`views.py` has defensive import handling (lines 24-42) but `normalization_views.py` doesn't

## Recommended Actions

### 1. **Choose a Single Normalization System**
Decide between:
- Django's `TextChoices` for database storage
- The Enum-based system for more complex normalization methods
- Or create a unified system that bridges both

### 2. **Fix `normalization_views.py`**
Replace incorrect usage:
```python
# Current (incorrect)
normalizer = NumNorm(method)

# Should be either:
# Option 1: Use Normalizador
from ..normalization_methods import Normalizador, NumNorm, TextNorm
normalizador = Normalizador(metodo_numerico=NumNorm[method])

# Option 2: Direct function calls
from ..normalization_methods import DISPATCH_NUM, NumNorm
func = DISPATCH_NUM[NumNorm[method]]
normalized_values = func(df[column])
```

### 3. **Remove Unused Imports**
Remove `Normalizador` import from `normalization_views.py` if not using it

### 4. **Standardize Method Names**
Create a mapping between frontend strings and backend enums:
```python
METHOD_MAPPING = {
    'min_max': NumNorm.MIN_MAX,
    'z_score': NumNorm.Z_SCORE,
    'onehot': TextNorm.ONE_HOT,
    # etc.
}
```

### 5. **Update Database Schema**
The `TrainingSession.normalization_method` field uses Django choices while `Dataset.normalization_method` is a free-form string

## Files Affected
1. `/ml_trainer/views/normalization_views.py` - Main issue location
2. `/ml_trainer/normalization_methods.py` - Contains unused `Normalizador`
3. `/ml_trainer/views.py` - Has correct implementation
4. `/ml_trainer/models.py` - Different normalization choices
5. `/ml_trainer/ml_utils.py` - String-based normalization