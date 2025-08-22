"""
Módulo 2: Configuración de Ejecución
Maneja la configuración de cómo se ejecutará el entrenamiento del modelo
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterator, Tuple
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, LeaveOneOut, RepeatedKFold, RepeatedStratifiedKFold


class ExecutionStrategy(ABC):
    """Estrategia base para la configuración de ejecución del modelo"""
    
    @abstractmethod
    def get_splits(self, X, y, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Obtiene los splits para entrenamiento y validación"""
        pass
    
    @abstractmethod
    def get_n_splits(self) -> int:
        """Retorna el número de splits que generará"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Retorna una descripción de la estrategia"""
        pass


class StandardExecution(ExecutionStrategy):
    """Ejecución estándar sin cross-validation (usa la división de Module 1)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def get_splits(self, X, y, groups=None):
        # Esta estrategia no genera splits, usa la división del Module 1
        yield None, None
    
    def get_n_splits(self):
        return 1
    
    def get_description(self):
        return "Ejecución estándar sin cross-validation"


class KFoldExecution(ExecutionStrategy):
    """Ejecución con K-Fold Cross Validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_splits = config.get('n_splits', 5)
        self.shuffle = config.get('shuffle', True)
        self.random_state = config.get('random_state', None)
        self.cv_train_size = config.get('cv_train_size', 0.8)
        self.cv_val_size = config.get('cv_val_size', 0.2)
        
        self.kfold = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
    
    def get_splits(self, X, y, groups=None):
        return self.kfold.split(X, y, groups)
    
    def get_n_splits(self):
        return self.n_splits
    
    def get_description(self):
        return f"K-Fold Cross Validation con K={self.n_splits}"


class StratifiedKFoldExecution(ExecutionStrategy):
    """Ejecución con Stratified K-Fold Cross Validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_splits = config.get('n_splits', 5)
        self.shuffle = config.get('shuffle', True)
        self.random_state = config.get('random_state', None)
        self.cv_train_size = config.get('cv_train_size', 0.8)
        self.cv_val_size = config.get('cv_val_size', 0.2)
        
        self.kfold = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
    
    def get_splits(self, X, y, groups=None):
        return self.kfold.split(X, y, groups)
    
    def get_n_splits(self):
        return self.n_splits
    
    def get_description(self):
        return f"Stratified K-Fold Cross Validation con K={self.n_splits}"


class TimeSeriesSplitExecution(ExecutionStrategy):
    """Ejecución con Time Series Split para datos temporales"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_splits = config.get('n_splits', 5)
        self.max_train_size = config.get('max_train_size', None)
        self.test_size = config.get('test_size', None)
        self.gap = config.get('gap', 0)
        
        self.tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            gap=self.gap
        )
    
    def get_splits(self, X, y, groups=None):
        return self.tscv.split(X, y, groups)
    
    def get_n_splits(self):
        return self.n_splits
    
    def get_description(self):
        return f"Time Series Split con {self.n_splits} splits"


class LeaveOneOutExecution(ExecutionStrategy):
    """Ejecución con Leave-One-Out Cross Validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loo = LeaveOneOut()
    
    def get_splits(self, X, y, groups=None):
        return self.loo.split(X, y, groups)
    
    def get_n_splits(self):
        # Este método debe ser llamado con los datos reales
        # ya que LOO crea tantos folds como muestras hay
        return -1  # Indicador especial para LOO
    
    def get_description(self):
        return "Leave-One-Out Cross Validation"


class RepeatedKFoldExecution(ExecutionStrategy):
    """Ejecución con Repeated K-Fold Cross Validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_splits = config.get('n_splits', 5)
        self.n_repeats = config.get('n_repeats', 10)
        self.random_state = config.get('random_state', None)
        self.cv_train_size = config.get('cv_train_size', 0.8)
        self.cv_val_size = config.get('cv_val_size', 0.2)
        
        self.rkf = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )
    
    def get_splits(self, X, y, groups=None):
        return self.rkf.split(X, y, groups)
    
    def get_n_splits(self):
        return self.n_splits * self.n_repeats
    
    def get_description(self):
        return f"Repeated K-Fold CV con K={self.n_splits}, {self.n_repeats} repeticiones"


class RepeatedStratifiedKFoldExecution(ExecutionStrategy):
    """Ejecución con Repeated Stratified K-Fold Cross Validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_splits = config.get('n_splits', 5)
        self.n_repeats = config.get('n_repeats', 10)
        self.random_state = config.get('random_state', None)
        self.cv_train_size = config.get('cv_train_size', 0.8)
        self.cv_val_size = config.get('cv_val_size', 0.2)
        
        self.rskf = RepeatedStratifiedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )
    
    def get_splits(self, X, y, groups=None):
        return self.rskf.split(X, y, groups)
    
    def get_n_splits(self):
        return self.n_splits * self.n_repeats
    
    def get_description(self):
        return f"Repeated Stratified K-Fold CV con K={self.n_splits}, {self.n_repeats} repeticiones"


class ExecutionConfigManager:
    """Gestor principal para la configuración de ejecución"""
    
    STRATEGIES = {
        'standard': StandardExecution,
        'kfold': KFoldExecution,
        'stratified_kfold': StratifiedKFoldExecution,
        'time_series_split': TimeSeriesSplitExecution,
        'leave_one_out': LeaveOneOutExecution,
        'repeated_kfold': RepeatedKFoldExecution,
        'repeated_stratified_kfold': RepeatedStratifiedKFoldExecution
    }
    
    @classmethod
    def create_execution_strategy(cls, strategy: str, config: Dict[str, Any]) -> ExecutionStrategy:
        """
        Crea una estrategia de ejecución basada en el tipo especificado
        
        Args:
            strategy: Tipo de estrategia ('standard', 'kfold', etc.)
            config: Configuración específica de la estrategia
            
        Returns:
            ExecutionStrategy: Instancia de la estrategia de ejecución
        """
        if strategy not in cls.STRATEGIES:
            raise ValueError(f"Estrategia de ejecución no soportada: {strategy}")
        
        strategy_class = cls.STRATEGIES[strategy]
        return strategy_class(config)
    
    @classmethod
    def get_available_strategies(cls) -> Dict[str, str]:
        """Retorna las estrategias disponibles con sus descripciones"""
        return {
            'standard': 'Ejecución Estándar (sin cross-validation)',
            'kfold': 'K-Fold Cross Validation',
            'stratified_kfold': 'Stratified K-Fold Cross Validation',
            'time_series_split': 'Time Series Split',
            'leave_one_out': 'Leave-One-Out Cross Validation',
            'repeated_kfold': 'Repeated K-Fold Cross Validation',
            'repeated_stratified_kfold': 'Repeated Stratified K-Fold Cross Validation'
        }