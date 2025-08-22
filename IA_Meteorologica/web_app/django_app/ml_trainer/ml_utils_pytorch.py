"""
PyTorch implementation of neural network models
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os


class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Get last time step output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and final layer
        out = self.dropout(last_output)
        out = self.fc(out)
        
        return out


class GRUModel(nn.Module):
    """GRU model for time series prediction"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Get last time step output
        last_output = gru_out[:, -1, :]
        
        # Apply dropout and final layer
        out = self.dropout(last_output)
        out = self.fc(out)
        
        return out


class CNNModel(nn.Module):
    """CNN model for time series prediction"""
    def __init__(self, input_channels, sequence_length, output_size, 
                 filters=64, kernel_size=3, dropout=0.2):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, filters, kernel_size, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(filters, filters*2, kernel_size, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc = nn.Linear(filters*2, output_size)
        
    def forward(self, x):
        # Reshape for Conv1d: (batch, features, sequence)
        x = x.transpose(1, 2)
        
        # First conv block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten and final layer
        x = x.squeeze(-1)
        x = self.fc(x)
        
        return x


class TransformerModel(nn.Module):
    """Transformer model for time series prediction"""
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Final prediction
        x = self.fc(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class CustomModel(nn.Module):
    """Custom model built from architecture JSON"""
    def __init__(self, architecture, input_shape, output_shape):
        super(CustomModel, self).__init__()
        
        self.layers = nn.ModuleList()
        prev_size = input_shape[-1]  # Last dimension is feature size
        
        for layer_config in architecture:
            layer_type = layer_config.get('type')
            params = layer_config.get('params', {})
            
            if layer_type == 'Dense':
                units = params.get('units', 32)
                self.layers.append(nn.Linear(prev_size, units))
                
                # Add activation
                activation = params.get('activation', 'relu')
                if activation == 'relu':
                    self.layers.append(nn.ReLU())
                elif activation == 'tanh':
                    self.layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    self.layers.append(nn.Sigmoid())
                elif activation == 'elu':
                    self.layers.append(nn.ELU())
                
                prev_size = units
                
            elif layer_type == 'LSTM':
                units = params.get('units', 50)
                num_layers = params.get('num_layers', 1)
                dropout = params.get('dropout', 0.0)
                
                lstm = nn.LSTM(
                    prev_size, 
                    units, 
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.layers.append(lstm)
                self.layers.append(LSTMExtractor())  # Custom layer to extract last output
                prev_size = units
                
            elif layer_type == 'GRU':
                units = params.get('units', 50)
                num_layers = params.get('num_layers', 1)
                dropout = params.get('dropout', 0.0)
                
                gru = nn.GRU(
                    prev_size, 
                    units, 
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.layers.append(gru)
                self.layers.append(GRUExtractor())  # Custom layer to extract last output
                prev_size = units
                
            elif layer_type == 'Dropout':
                rate = params.get('rate', 0.5)
                self.layers.append(nn.Dropout(rate))
                
            elif layer_type == 'BatchNormalization':
                self.layers.append(nn.BatchNorm1d(prev_size))
                
            elif layer_type == 'Conv1D':
                filters = params.get('filters', 32)
                kernel_size = params.get('kernel_size', 3)
                stride = params.get('strides', 1)
                padding = params.get('padding', 'same')
                
                if padding == 'same':
                    padding = kernel_size // 2
                elif padding == 'valid':
                    padding = 0
                
                self.layers.append(nn.Conv1d(prev_size, filters, kernel_size, stride, padding))
                
                # Add activation
                activation = params.get('activation', 'relu')
                if activation == 'relu':
                    self.layers.append(nn.ReLU())
                elif activation == 'tanh':
                    self.layers.append(nn.Tanh())
                
                prev_size = filters
        
        # Add final output layer if needed
        if prev_size != output_shape[0]:
            self.layers.append(nn.Linear(prev_size, output_shape[0]))
    
    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, (nn.LSTM, nn.GRU)):
                x, _ = layer(x)
            elif isinstance(layer, (LSTMExtractor, GRUExtractor)):
                x = layer(x)
            else:
                x = layer(x)
        return x


class LSTMExtractor(nn.Module):
    """Extract last output from LSTM"""
    def forward(self, x):
        return x[:, -1, :]


class GRUExtractor(nn.Module):
    """Extract last output from GRU"""
    def forward(self, x):
        return x[:, -1, :]


def get_optimizer_pytorch(optimizer_name, model_parameters, learning_rate):
    """Get PyTorch optimizer from string name"""
    optimizers = {
        'adam': optim.Adam(model_parameters, lr=learning_rate),
        'sgd': optim.SGD(model_parameters, lr=learning_rate, momentum=0.9),
        'rmsprop': optim.RMSprop(model_parameters, lr=learning_rate),
        'adagrad': optim.Adagrad(model_parameters, lr=learning_rate),
        'adadelta': optim.Adadelta(model_parameters, lr=learning_rate),
        'adamax': optim.Adamax(model_parameters, lr=learning_rate),
        'adamw': optim.AdamW(model_parameters, lr=learning_rate)
    }
    return optimizers.get(optimizer_name.lower(), optim.Adam(model_parameters, lr=learning_rate))


def get_loss_function(loss_name):
    """Get PyTorch loss function from string name"""
    losses = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'huber': nn.HuberLoss(),
        'smooth_l1': nn.SmoothL1Loss()
    }
    return losses.get(loss_name.lower(), nn.MSELoss())


def build_pytorch_model(model_type, input_shape, output_shape, hyperparams, custom_architecture=None):
    """Build PyTorch model based on type"""
    
    input_size = input_shape[-1]  # Number of features
    output_size = output_shape[0]
    
    if custom_architecture:
        model = CustomModel(custom_architecture, input_shape, output_shape)
    elif model_type == 'lstm':
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hyperparams.get('units', 50),
            num_layers=hyperparams.get('layers', 2),
            output_size=output_size,
            dropout=hyperparams.get('dropout', 0.2)
        )
    elif model_type == 'gru':
        model = GRUModel(
            input_size=input_size,
            hidden_size=hyperparams.get('units', 50),
            num_layers=hyperparams.get('layers', 2),
            output_size=output_size,
            dropout=hyperparams.get('dropout', 0.2)
        )
    elif model_type == 'cnn':
        model = CNNModel(
            input_channels=input_size,
            sequence_length=input_shape[0],
            output_size=output_size,
            filters=hyperparams.get('filters', 64),
            kernel_size=hyperparams.get('kernel_size', 3),
            dropout=hyperparams.get('dropout', 0.2)
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            input_size=input_size,
            d_model=hyperparams.get('d_model', 128),
            nhead=hyperparams.get('nhead', 8),
            num_layers=hyperparams.get('num_layers', 3),
            output_size=output_size,
            dropout=hyperparams.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unsupported model type for PyTorch: {model_type}")
    
    return model


def train_pytorch_model(session, model, X_train, y_train, X_val, y_val, hyperparams):
    """Train PyTorch model"""
    
    # Set random seeds for reproducibility
    if hasattr(session, 'random_state') and session.random_state is not None:
        torch.manual_seed(session.random_state)
        torch.cuda.manual_seed_all(session.random_state)
        np.random.seed(session.random_state)
        print(f"[train_pytorch_model] Using global random_state: {session.random_state}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    batch_size = hyperparams.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Only create validation tensors and loader if validation data exists
    has_validation = len(X_val) > 0
    if has_validation:
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Setup optimizer and loss
    optimizer = get_optimizer_pytorch(
        hyperparams.get('optimizer', 'adam'),
        model.parameters(),
        hyperparams.get('learning_rate', 0.001)
    )
    
    criterion = get_loss_function(hyperparams.get('loss', 'mse'))
    
    # Training loop
    epochs = hyperparams.get('epochs', 50)
    history = {'train_loss': [], 'val_loss': []}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = model.state_dict()  # Initialize with current state
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase only if validation data exists
        if has_validation:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                # Restore best model
                model.load_state_dict(best_model_state)
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            # No validation, just print training loss
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
    
    return model, history


def save_pytorch_model(model, filepath):
    """Save PyTorch model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }, filepath)


def load_pytorch_model(filepath, model_class, *args, **kwargs):
    """Load PyTorch model"""
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = model_class(*args, **kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model