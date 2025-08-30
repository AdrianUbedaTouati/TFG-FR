"""
Módulo de División de Datos - Global para todos los modelos

Este módulo proporciona diferentes estrategias de división de datos que pueden ser
utilizadas por cualquier tipo de modelo (Neural Networks, Random Forest, XGBoost, etc.)

Estrategias soportadas:
- Aleatoria: División aleatoria simple
- Estratificada: Mantiene proporciones de clases en clasificación
- Por grupos: No mezcla grupos entre train/val/test
- Temporal: Para series de tiempo (holdout, blocked, walk-forward)
- Secuencial: Sin reordenamiento, mantiene orden original de entrada
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, 
    StratifiedShuffleSplit,
    GroupShuffleSplit,
    TimeSeriesSplit,
    GroupKFold,
    StratifiedKFold
)
from typing import Tuple, Optional, Union, Dict, Any, List
from abc import ABC, abstractmethod


class DataSplitStrategy(ABC):
    """Clase base abstracta para estrategias de división de datos"""
    
    def __init__(self, train_size: float, val_size: float, test_size: float, 
                 random_state: Optional[int] = None):
        """
        Args:
            train_size: Proporción de datos para entrenamiento (0-1)
            val_size: Proporción de datos para validación (0-1)
            test_size: Proporción de datos para test (0-1)
            random_state: Semilla para reproducibilidad
        """
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        
        # Validar que las proporciones sumen 1
        total = train_size + val_size + test_size
        
        # Permitir un margen más amplio para errores de precisión flotante
        if not (0.98 <= total <= 1.02):  # Margen de ±2%
            raise ValueError(f"Las proporciones deben sumar 1.0, actualmente suman {total:.4f}")
        
        # Normalizar las proporciones para que sumen exactamente 1.0
        if total != 1.0:
            self.train_size = train_size / total
            self.val_size = val_size / total
            self.test_size = test_size / total
        else:
            self.train_size = train_size
            self.val_size = val_size
            self.test_size = test_size
    
    @abstractmethod
    def split(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Divide los datos en train, validation y test
        
        Returns:
            Tupla de (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        pass
    
    def get_indices(self, n_samples: int, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Obtiene los índices para train, val y test sin necesidad de los datos
        
        Returns:
            Tupla de (train_indices, val_indices, test_indices)
        """
        # Por defecto, crear datos dummy y obtener índices
        X_dummy = np.zeros((n_samples, 1))
        y_dummy = np.zeros(n_samples)
        
        # Usar split para obtener los datos divididos
        X_train, y_train, X_val, y_val, X_test, y_test = self.split(X_dummy, y_dummy, **kwargs)
        
        # Calcular índices basándose en los tamaños
        train_size = len(X_train)
        val_size = len(X_val)
        
        train_indices = np.arange(train_size)
        val_indices = np.arange(train_size, train_size + val_size)
        test_indices = np.arange(train_size + val_size, n_samples)
        
        return train_indices, val_indices, test_indices


class RandomSplitStrategy(DataSplitStrategy):
    """División aleatoria simple de los datos"""
    
    def split(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """División aleatoria con mezcla de datos"""
        
        # Primero separar train+val de test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )
        
        # Si no hay validación (val_size = 0), no hacer segunda división
        if self.val_size == 0:
            X_train, y_train = X_temp, y_temp
            # Crear arrays vacíos para validación manteniendo las dimensiones correctas
            X_val = np.array([]).reshape(0, X.shape[1]) if len(X.shape) > 1 else np.array([])
            y_val = np.array([]).reshape(0, y.shape[1]) if len(y.shape) > 1 else np.array([])
        else:
            # Luego separar train de val
            val_proportion = self.val_size / (self.train_size + self.val_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_proportion,
                random_state=self.random_state,
                shuffle=True
            )
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def get_indices(self, n_samples: int, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Obtiene índices aleatorios para la división"""
        indices = np.arange(n_samples)
        np.random.RandomState(self.random_state).shuffle(indices)
        
        train_end = int(n_samples * self.train_size)
        val_end = int(n_samples * (self.train_size + self.val_size))
        
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]


class StratifiedSplitStrategy(DataSplitStrategy):
    """División estratificada para mantener proporciones de clases"""
    
    def split(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """División estratificada para clasificación"""
        
        # Para problemas multi-output, usar la primera columna para estratificar
        y_stratify = y if len(y.shape) == 1 else y[:, 0]
        
        try:
            # Primero separar train+val de test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_stratify if len(y.shape) == 1 else y_stratify[:-int(len(y)*self.test_size)],
                shuffle=True
            )
            
            # Si no hay validación (val_size = 0), no hacer segunda división
            if self.val_size == 0:
                X_train, y_train = X_temp, y_temp
                # Crear arrays vacíos para validación manteniendo las dimensiones correctas
                X_val = np.array([]).reshape(0, X.shape[1]) if len(X.shape) > 1 else np.array([])
                y_val = np.array([]).reshape(0, y.shape[1]) if len(y.shape) > 1 else np.array([])
            else:
                # Luego separar train de val
                val_proportion = self.val_size / (self.train_size + self.val_size)
                y_temp_stratify = y_temp if len(y_temp.shape) == 1 else y_temp[:, 0]
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp,
                    test_size=val_proportion,
                    random_state=self.random_state,
                    stratify=y_temp_stratify,
                    shuffle=True
                )
        except ValueError as e:
            # Si la estratificación falla (ej: muy pocas muestras por clase), 
            # usar división aleatoria como fallback
            print(f"Advertencia: Estratificación falló ({str(e)}), usando división aleatoria")
            random_strategy = RandomSplitStrategy(
                self.train_size, self.val_size, self.test_size, self.random_state
            )
            return random_strategy.split(X, y)
        
        return X_train, y_train, X_val, y_val, X_test, y_test


class GroupSplitStrategy(DataSplitStrategy):
    """División por grupos para evitar filtración de datos entre grupos"""
    
    def split(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, **kwargs) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """División manteniendo grupos completos en cada conjunto"""
        
        if groups is None:
            raise ValueError("Se requiere el parámetro 'groups' para división por grupos")
        
        # Obtener grupos únicos
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        # Mezclar grupos aleatoriamente
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(unique_groups)
        
        # Calcular número de grupos para cada conjunto
        n_train_groups = int(n_groups * self.train_size)
        n_val_groups = int(n_groups * self.val_size)
        
        # Asignar grupos a cada conjunto
        train_groups = unique_groups[:n_train_groups]
        val_groups = unique_groups[n_train_groups:n_train_groups + n_val_groups]
        test_groups = unique_groups[n_train_groups + n_val_groups:]
        
        # Crear máscaras para cada conjunto
        train_mask = np.isin(groups, train_groups)
        val_mask = np.isin(groups, val_groups)
        test_mask = np.isin(groups, test_groups)
        
        # Dividir datos según las máscaras
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        return X_train, y_train, X_val, y_val, X_test, y_test


class TemporalSplitStrategy(DataSplitStrategy):
    """División temporal para series de tiempo"""
    
    def __init__(self, train_size: float, val_size: float, test_size: float,
                 method: str = 'holdout', gap: int = 0, 
                 n_splits: Optional[int] = None, random_state: Optional[int] = None):
        """
        Args:
            method: 'holdout', 'blocked' o 'walk_forward'
            gap: Número de periodos de separación entre train y test
            n_splits: Número de divisiones para walk-forward validation
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.method = method
        self.gap = gap
        self.n_splits = n_splits or 5
    
    def split(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """División temporal según el método especificado"""
        
        n_samples = len(X)
        
        if self.method == 'holdout':
            # División simple temporal: train|val|test en orden cronológico
            train_end = int(n_samples * self.train_size)
            val_end = int(n_samples * (self.train_size + self.val_size))
            
            # Aplicar gap si está especificado
            if self.gap > 0:
                train_end -= self.gap
                val_end -= self.gap
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[train_end + self.gap:val_end]
            y_val = y[train_end + self.gap:val_end]
            X_test = X[val_end + self.gap:]
            y_test = y[val_end + self.gap:]
            
        elif self.method == 'blocked':
            # División en bloques temporales con posible mezcla de bloques
            # Útil cuando hay patrones estacionales
            block_size = n_samples // 10  # Dividir en 10 bloques por defecto
            n_blocks = n_samples // block_size
            
            blocks = np.arange(n_blocks)
            if self.random_state is not None:
                rng = np.random.RandomState(self.random_state)
                rng.shuffle(blocks)
            
            n_train_blocks = int(n_blocks * self.train_size)
            n_val_blocks = int(n_blocks * self.val_size)
            
            train_blocks = blocks[:n_train_blocks]
            val_blocks = blocks[n_train_blocks:n_train_blocks + n_val_blocks]
            test_blocks = blocks[n_train_blocks + n_val_blocks:]
            
            # Crear índices para cada conjunto
            train_indices = []
            val_indices = []
            test_indices = []
            
            for block in train_blocks:
                start = block * block_size
                end = min((block + 1) * block_size, n_samples)
                train_indices.extend(range(start, end))
            
            for block in val_blocks:
                start = block * block_size
                end = min((block + 1) * block_size, n_samples)
                val_indices.extend(range(start, end))
                
            for block in test_blocks:
                start = block * block_size
                end = min((block + 1) * block_size, n_samples)
                test_indices.extend(range(start, end))
            
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_val = X[val_indices]
            y_val = y[val_indices]
            X_test = X[test_indices]
            y_test = y[test_indices]
            
        elif self.method == 'walk_forward':
            # Walk-forward validation: múltiples train/test splits expandiéndose
            # Para simplicidad, retornamos el último split
            tscv = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)
            
            # Obtener el último split
            for train_val_idx, test_idx in tscv.split(X):
                pass  # Quedarnos con el último
            
            # Dividir train_val en train y val
            n_train_val = len(train_val_idx)
            n_train = int(n_train_val * (self.train_size / (self.train_size + self.val_size)))
            
            train_idx = train_val_idx[:n_train]
            val_idx = train_val_idx[n_train:]
            
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            
        else:
            raise ValueError(f"Método temporal no reconocido: {self.method}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test


class SequentialSplitStrategy(DataSplitStrategy):
    """División secuencial que mantiene el orden original de los datos"""
    
    def split(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """División secuencial sin reordenamiento de datos"""
        
        n_samples = len(X)
        
        # Calcular índices de corte basados en las proporciones
        train_end = int(n_samples * self.train_size)
        val_end = int(n_samples * (self.train_size + self.val_size))
        
        # Dividir según orden de entrada (sin shuffling)
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def get_indices(self, n_samples: int, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Obtiene índices secuenciales para la división"""
        train_end = int(n_samples * self.train_size)
        val_end = int(n_samples * (self.train_size + self.val_size))
        
        train_indices = np.arange(0, train_end)
        val_indices = np.arange(train_end, val_end)
        test_indices = np.arange(val_end, n_samples)
        
        return train_indices, val_indices, test_indices


class DataSplitter:
    """Clase principal para manejar la división de datos"""
    
    STRATEGIES = {
        'random': RandomSplitStrategy,
        'stratified': StratifiedSplitStrategy,
        'group': GroupSplitStrategy,
        'temporal': TemporalSplitStrategy,
        'sequential': SequentialSplitStrategy
    }
    
    @classmethod
    def create_splitter(cls, strategy: str, config: Dict[str, Any]) -> DataSplitStrategy:
        """
        Factory method para crear la estrategia de división apropiada
        
        Args:
            strategy: Nombre de la estrategia ('random', 'stratified', 'group', 'temporal')
            config: Configuración específica para la estrategia
                - train_size: float
                - val_size: float
                - test_size: float
                - random_state: int (opcional)
                - method: str (para temporal)
                - gap: int (para temporal)
                - n_splits: int (para temporal walk-forward)
        
        Returns:
            Instancia de DataSplitStrategy
        """
        if strategy not in cls.STRATEGIES:
            raise ValueError(f"Estrategia no reconocida: {strategy}. "
                           f"Opciones válidas: {list(cls.STRATEGIES.keys())}")
        
        strategy_class = cls.STRATEGIES[strategy]
        
        # Extraer parámetros comunes
        train_size = config.get('train_size', 0.7)
        val_size = config.get('val_size', 0.15)
        test_size = config.get('test_size', 0.15)
        random_state = config.get('random_state', None)
        
        # Crear instancia según el tipo
        if strategy == 'temporal':
            return strategy_class(
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                method=config.get('method', 'holdout'),
                gap=config.get('gap', 0),
                n_splits=config.get('n_splits', 5),
                random_state=random_state
            )
        else:
            return strategy_class(
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                random_state=random_state
            )
    
    @staticmethod
    def validate_config(strategy: str, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Valida la configuración para una estrategia dada
        
        Returns:
            Tupla de (es_válido, mensaje_error)
        """
        # Validar proporciones
        train_size = config.get('train_size', 0.7)
        val_size = config.get('val_size', 0.15)
        test_size = config.get('test_size', 0.15)
        
        total = train_size + val_size + test_size
        # Permitir un margen más amplio para errores de precisión flotante
        if not (0.98 <= total <= 1.02):  # Margen de ±2%
            return False, f"Las proporciones deben sumar 1.0, actualmente suman {total:.4f}"
        
        if any(size < 0 or size > 1 for size in [train_size, val_size, test_size]):
            return False, "Todas las proporciones deben estar entre 0 y 1"
        
        # Validaciones específicas por estrategia
        if strategy == 'group' and 'group_column' not in config:
            return False, "La estrategia 'group' requiere especificar 'group_column'"
        
        if strategy == 'temporal':
            method = config.get('method', 'holdout')
            if method not in ['holdout', 'blocked', 'walk_forward']:
                return False, f"Método temporal inválido: {method}"
        
        return True, None
    
    @staticmethod
    def get_strategy_info(strategy: str) -> Dict[str, Any]:
        """Obtiene información sobre una estrategia específica"""
        info = {
            'random': {
                'name': 'División Aleatoria',
                'description': 'Divide los datos de forma aleatoria, mezclando las muestras',
                'use_cases': ['Datos i.i.d.', 'Clasificación general', 'Regresión general'],
                'params': ['train_size', 'val_size', 'test_size', 'random_state']
            },
            'stratified': {
                'name': 'División Estratificada',
                'description': 'Mantiene la proporción de clases en cada conjunto',
                'use_cases': ['Clasificación con clases desbalanceadas', 'Datos con distribución importante'],
                'params': ['train_size', 'val_size', 'test_size', 'random_state'],
                'requirements': ['Solo para problemas de clasificación']
            },
            'group': {
                'name': 'División por Grupos',
                'description': 'Mantiene grupos completos en cada conjunto (no mezcla)',
                'use_cases': ['Datos agrupados por usuario/sesión', 'Evitar filtración entre grupos'],
                'params': ['train_size', 'val_size', 'test_size', 'group_column', 'random_state'],
                'requirements': ['Columna de grupos en el dataset']
            },
            'temporal': {
                'name': 'División Temporal',
                'description': 'Respeta el orden temporal de los datos',
                'use_cases': ['Series de tiempo', 'Datos con dependencia temporal'],
                'params': ['train_size', 'val_size', 'test_size', 'method', 'gap', 'n_splits'],
                'methods': {
                    'holdout': 'División simple en orden cronológico',
                    'blocked': 'División en bloques temporales',
                    'walk_forward': 'Validación walk-forward expanding window'
                }
            },
            'sequential': {
                'name': 'División Secuencial',
                'description': 'Mantiene el orden original de entrada de los datos sin reordenamiento',
                'use_cases': ['Datos ordenados por importancia', 'Datasets cronológicos simples', 'Cuando el orden original es significativo'],
                'params': ['train_size', 'val_size', 'test_size'],
                'note': 'No utiliza random_state ya que no hay aleatorización'
            }
        }
        
        return info.get(strategy, {})