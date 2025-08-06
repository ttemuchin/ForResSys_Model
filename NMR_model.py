import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DynamicNMRDataset(Dataset):
    def __init__(self, *x_signals, y):
        """
        :param x_signals: Списки сигналов (каждый размером [P, L_i], L_i может отличаться)
        :param y: Целевые переменные [P, N]
        """
        self.x_signals = [torch.FloatTensor(x) for x in x_signals]
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return tuple(x[idx] for x in self.x_signals) + (self.y[idx],)

class DynamicNMRRegressor(nn.Module):
    def __init__(self, input_dims: list, num_targets: int, conv_filters: int = 32):
        """
        :param input_dims: Список размерностей для каждого типа сигнала (например, [1000, 2000] для FID и CPMG)
        :param num_targets: Количество целевых переменных (N)
        :param conv_filters: Базовое количество фильтров в сверточных слоях
        """
        super().__init__()
        self.num_experiments = len(input_dims)  # M
        self.num_targets = num_targets          # N
        
        # Динамическое создание ветвей для каждого типа сигнала
        self.branches = nn.ModuleList()
        for dim in input_dims:
            branch = nn.Sequential(
                nn.Conv1d(1, conv_filters, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(4 if dim >= 2000 else 2),
                nn.Conv1d(conv_filters, conv_filters * 2, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Flatten()
            )
            self.branches.append(branch)
        
        # Вычисление общего размера признаков после всех ветвей
        self.total_features = 0
        for i, dim in enumerate(input_dims):
            dummy_input = torch.zeros(1, 1, dim)
            flattened_size = self.branches[i](dummy_input).shape[1]
            self.total_features += flattened_size
        
        # Финальный классификатор
        self.final_fc = nn.Sequential(
            nn.Linear(self.total_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_targets)
        )
    def forward(self, *x_signals):
        # Обработка каждого сигнала через свою ветвь
        features = []
        for i in range(self.num_experiments):
            x = x_signals[i].unsqueeze(1)  # Добавляем размерность канала [B, 1, L]
            features.append(self.branches[i](x))
        
        # Объединение всех признаков
        combined = torch.cat(features, dim=1)
        return self.final_fc(combined)