"""
Script para reentrenar los modelos especialistas con datasets corregidos
que incluyen TODAS las clases para manejar muestras fuera de dominio
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar el directorio padre al path para importar módulos
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

class WeatherDataset(Dataset):
    """Dataset personalizado para datos meteorológicos"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMLineClassifier(nn.Module):
    """Modelo LSTM para clasificación de series temporales"""
    def __init__(self, in_features, num_classes, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.in_features = in_features
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(
            in_features, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Si x es 2D, añadir dimensión temporal
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Usar la última salida
        out = lstm_out[:, -1, :]
        
        # Capas densas
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class SpecialistTrainer:
    """Entrenador para modelos especialistas con manejo de clases fuera de dominio"""
    
    def __init__(self, specialist_name: str, config: Dict):
        self.specialist_name = specialist_name
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Configurar logging
        self.setup_logging()
        
        # Cargar configuración de pesos de clase
        self.class_weights = torch.tensor([
            float(config['class_weights']['0']),  # Clase objetivo 1
            float(config['class_weights']['1']),  # Clase objetivo 2
            float(config['class_weights']['2'])   # Clase "Other"
        ]).to(self.device)
        
        self.logger.info(f"Inicializando entrenador para {specialist_name}")
        self.logger.info(f"Dispositivo: {self.device}")
        self.logger.info(f"Pesos de clase: {self.class_weights.cpu().numpy()}")
    
    def setup_logging(self):
        """Configura el sistema de logging"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"retrain_{self.specialist_name}_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.specialist_name)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Carga y prepara los datos corregidos"""
        data_path = self.config['data_path']
        self.logger.info(f"Cargando datos desde: {data_path}")
        
        df = pd.read_csv(data_path)
        self.logger.info(f"Forma del dataset: {df.shape}")
        
        # Identificar columnas de características y etiquetas
        feature_cols = [col for col in df.columns 
                       if not col.startswith('Weather Type') 
                       and col != 'Weather Type']
        
        label_cols = [col for col in df.columns if col.startswith('Weather Type_')]
        
        # Preparar X e y
        X = df[feature_cols].values.astype(np.float32)
        y = df[label_cols].values.argmax(axis=1)
        
        self.logger.info(f"Características: {len(feature_cols)}")
        self.logger.info(f"Clases: {label_cols}")
        self.logger.info(f"Distribución de clases: {np.bincount(y)}")
        
        return X, y, feature_cols, label_cols
    
    def create_weighted_sampler(self, y_train: np.ndarray) -> WeightedRandomSampler:
        """Crea un sampler con pesos para balancear el entrenamiento"""
        class_counts = np.bincount(y_train)
        
        # Calcular pesos inversamente proporcionales a la frecuencia
        # pero ajustados por los pesos de configuración
        sample_weights = np.zeros(len(y_train))
        
        for class_idx in range(len(class_counts)):
            mask = y_train == class_idx
            if class_idx == 2:  # Clase "Other"
                # Reducir la frecuencia de muestreo para la clase "Other"
                weight = 1.0 / (class_counts[class_idx] * 3)  # Factor de reducción
            else:
                # Clases objetivo con peso normal
                weight = 1.0 / class_counts[class_idx]
            sample_weights[mask] = weight
        
        # Normalizar pesos
        sample_weights = sample_weights / sample_weights.sum()
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   num_epochs=100, batch_size=32, learning_rate=0.001):
        """Entrena el modelo especialista"""
        
        # Crear datasets
        train_dataset = WeatherDataset(X_train, y_train)
        val_dataset = WeatherDataset(X_val, y_val)
        
        # Crear sampler ponderado
        sampler = self.create_weighted_sampler(y_train)
        
        # Crear dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=sampler
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        # Inicializar modelo
        in_features = X_train.shape[1]
        num_classes = 3  # 2 clases objetivo + 1 clase "Other"
        
        model = LSTMLineClassifier(
            in_features=in_features,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=2,
            dropout=0.3
        ).to(self.device)
        
        # Configurar optimizador y loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Scheduler para learning rate
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Historia de entrenamiento
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_acc_target': [], 'val_acc_target': [],
            'train_acc_other': [], 'val_acc_other': []
        }
        
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        early_stop_patience = 20
        
        self.logger.info("Iniciando entrenamiento...")
        
        for epoch in range(num_epochs):
            # Fase de entrenamiento
            model.train()
            train_loss = 0
            train_correct = 0
            train_correct_target = 0
            train_correct_other = 0
            train_total_target = 0
            train_total_other = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == y_batch).sum().item()
                
                # Calcular precisión por tipo de clase
                target_mask = y_batch < 2
                other_mask = y_batch == 2
                
                if target_mask.sum() > 0:
                    train_correct_target += (predicted[target_mask] == y_batch[target_mask]).sum().item()
                    train_total_target += target_mask.sum().item()
                
                if other_mask.sum() > 0:
                    train_correct_other += (predicted[other_mask] == y_batch[other_mask]).sum().item()
                    train_total_other += other_mask.sum().item()
            
            # Fase de validación
            model.eval()
            val_loss = 0
            val_correct = 0
            val_correct_target = 0
            val_correct_other = 0
            val_total_target = 0
            val_total_other = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == y_batch).sum().item()
                    
                    # Calcular precisión por tipo de clase
                    target_mask = y_batch < 2
                    other_mask = y_batch == 2
                    
                    if target_mask.sum() > 0:
                        val_correct_target += (predicted[target_mask] == y_batch[target_mask]).sum().item()
                        val_total_target += target_mask.sum().item()
                    
                    if other_mask.sum() > 0:
                        val_correct_other += (predicted[other_mask] == y_batch[other_mask]).sum().item()
                        val_total_other += other_mask.sum().item()
            
            # Calcular métricas
            train_acc = train_correct / len(train_dataset)
            val_acc = val_correct / len(val_dataset)
            
            train_acc_target = train_correct_target / train_total_target if train_total_target > 0 else 0
            val_acc_target = val_correct_target / val_total_target if val_total_target > 0 else 0
            
            train_acc_other = train_correct_other / train_total_other if train_total_other > 0 else 0
            val_acc_other = val_correct_other / val_total_other if val_total_other > 0 else 0
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Guardar historia
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_acc_target'].append(train_acc_target)
            history['val_acc_target'].append(val_acc_target)
            history['train_acc_other'].append(train_acc_other)
            history['val_acc_other'].append(val_acc_other)
            
            # Actualizar scheduler
            scheduler.step(avg_val_loss)
            
            # Guardar mejor modelo
            if val_acc_target > best_val_acc:  # Usar precisión en clases objetivo
                best_val_acc = val_acc_target
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Log cada 10 epochs
            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch [{epoch}/{num_epochs}] - "
                    f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} - "
                    f"Acc: {train_acc:.4f}/{val_acc:.4f} - "
                    f"Target Acc: {train_acc_target:.4f}/{val_acc_target:.4f} - "
                    f"Other Acc: {train_acc_other:.4f}/{val_acc_other:.4f}"
                )
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                self.logger.info(f"Early stopping en epoch {epoch}")
                break
        
        # Restaurar mejor modelo
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, history
    
    def save_model(self, model: nn.Module, feature_cols: List[str], 
                  label_cols: List[str], history: Dict):
        """Guarda el modelo y metadatos"""
        output_dir = f"outputs/retrained_{self.specialist_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Preparar metadatos
        metadata = {
            'specialist_name': self.specialist_name,
            'target_classes': self.config['target_classes'],
            'num_classes': 3,
            'in_features': len(feature_cols),
            'feature_names': feature_cols,
            'class_names': label_cols,
            'class_weights': self.class_weights.cpu().tolist(),
            'best_val_acc': max(history['val_acc']),
            'best_val_acc_target': max(history['val_acc_target']),
            'training_date': datetime.now().isoformat()
        }
        
        # Guardar checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }
        
        model_path = os.path.join(output_dir, f"{self.specialist_name}_corrected.pt")
        torch.save(checkpoint, model_path)
        self.logger.info(f"Modelo guardado en: {model_path}")
        
        # Guardar historia
        history_df = pd.DataFrame(history)
        history_path = os.path.join(output_dir, "training_history.csv")
        history_df.to_csv(history_path, index=False)
        
        # Guardar metadatos por separado
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Crear gráficos de entrenamiento
        self.plot_training_history(history, output_dir)
        
        return model_path
    
    def plot_training_history(self, history: Dict, output_dir: str):
        """Genera gráficos del historial de entrenamiento"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy general
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Overall Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Accuracy clases objetivo
        ax3.plot(epochs, history['train_acc_target'], 'b-', label='Train Target Acc')
        ax3.plot(epochs, history['val_acc_target'], 'r-', label='Val Target Acc')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Target Classes Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Accuracy clase "Other"
        ax4.plot(epochs, history['train_acc_other'], 'b-', label='Train Other Acc')
        ax4.plot(epochs, history['val_acc_other'], 'r-', label='Val Other Acc')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Other Class Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Gráficos guardados en: {plot_path}")
    
    def evaluate_model(self, model: nn.Module, X_test: np.ndarray, 
                      y_test: np.ndarray, label_cols: List[str]):
        """Evalúa el modelo en el conjunto de prueba"""
        model.eval()
        test_dataset = WeatherDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calcular métricas
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Reporte de clasificación
        report = classification_report(
            all_labels, all_preds, 
            target_names=label_cols,
            output_dict=True
        )
        
        self.logger.info("\nReporte de clasificación:")
        self.logger.info(classification_report(
            all_labels, all_preds, 
            target_names=label_cols
        ))
        
        # Matriz de confusión
        cm = confusion_matrix(all_labels, all_preds)
        
        # Visualizar matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_cols, yticklabels=label_cols)
        plt.title(f'Confusion Matrix - {self.specialist_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        output_dir = f"outputs/retrained_{self.specialist_name}"
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return report, cm
    
    def run(self):
        """Ejecuta el proceso completo de reentrenamiento"""
        self.logger.info("="*80)
        self.logger.info(f"REENTRENANDO {self.specialist_name.upper()}")
        self.logger.info("="*80)
        
        # Cargar datos
        X, y, feature_cols, label_cols = self.load_data()
        
        # División de datos
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
        )
        
        self.logger.info(f"\nDivisión de datos:")
        self.logger.info(f"  - Train: {len(X_train)} muestras")
        self.logger.info(f"  - Val: {len(X_val)} muestras")
        self.logger.info(f"  - Test: {len(X_test)} muestras")
        
        # Entrenar modelo
        model, history = self.train_model(
            X_train, y_train, X_val, y_val,
            num_epochs=150,
            batch_size=32,
            learning_rate=0.001
        )
        
        # Guardar modelo
        model_path = self.save_model(model, feature_cols, label_cols, history)
        
        # Evaluar modelo
        report, cm = self.evaluate_model(model, X_test, y_test, label_cols)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("RESUMEN DEL REENTRENAMIENTO")
        self.logger.info("="*80)
        self.logger.info(f"Modelo: {self.specialist_name}")
        self.logger.info(f"Mejor precisión en validación (general): {max(history['val_acc']):.4f}")
        self.logger.info(f"Mejor precisión en validación (clases objetivo): {max(history['val_acc_target']):.4f}")
        self.logger.info(f"Precisión en test (general): {report['accuracy']:.4f}")
        self.logger.info(f"Modelo guardado en: {model_path}")
        
        return model_path, report

def main():
    """Función principal para reentrenar todos los especialistas"""
    print("="*80)
    print("REENTRENAMIENTO DE MODELOS ESPECIALISTAS")
    print("="*80)
    
    # Cargar configuración
    config_path = "data_corrected/training_config.json"
    if not os.path.exists(config_path):
        print(f"ERROR: No se encontró la configuración en {config_path}")
        print("Ejecute primero prepare_corrected_datasets.py")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    results = {}
    
    # Reentrenar cada especialista
    for specialist_key, specialist_config in config.items():
        specialist_name = specialist_key.replace("specialist_", "")
        print(f"\n{'='*60}")
        print(f"Procesando Especialista {specialist_name.upper()}")
        print(f"{'='*60}")
        
        trainer = SpecialistTrainer(specialist_name, specialist_config)
        model_path, report = trainer.run()
        
        results[specialist_name] = {
            'model_path': model_path,
            'accuracy': report['accuracy'],
            'report': report
        }
    
    # Guardar resumen de resultados
    summary_path = "outputs/retraining_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("REENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"\nResumen guardado en: {summary_path}")
    
    for specialist, result in results.items():
        print(f"\n{specialist.upper()}:")
        print(f"  - Accuracy: {result['accuracy']:.4f}")
        print(f"  - Modelo: {result['model_path']}")
    
    print("\nLos modelos reentrenados ahora pueden manejar muestras fuera de dominio.")
    print("Ejecute evaluate_architectures_corrected.py para evaluar el rendimiento mejorado.")

if __name__ == "__main__":
    main()