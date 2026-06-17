import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
# from torch.nn import functional as F

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

class ConvolutionalRegressor(nn.Module):
    def __init__(self, input_dims: list, num_targets: int, conv_filters: int = 32):
        """
        :param input_dims: Список длин сигналов (например, [1000, 2000])
        :param num_targets: Количество целевых переменных (N)
        :param conv_filters: Базовое число фильтров в сверточных слоях
        """
        super().__init__()
        self.num_experiments = len(input_dims)
        self.num_targets = num_targets
        
        # 1. Подготовка к объединению сигналов
        self.max_len = max(input_dims)
        
        # 2. Общие сверточные слои
        self.shared_conv = nn.Sequential(
            nn.Conv1d(self.num_experiments, conv_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_filters, conv_filters * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        # 3. Вычисление размера выхода сверточной части
        dummy_input = torch.zeros(1, self.num_experiments, self.max_len)
        conv_out_size = self.shared_conv(dummy_input).shape[1]
        
        # 4. Финальные слои
        self.final_fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_targets)
        )

    def forward(self, *x_signals):
        # 1. Приведение сигналов к одинаковой длине
        padded_signals = []
        for x in x_signals:
            # padding к каждому сигналу
            pad_size = self.max_len - x.shape[-1]
            padded = torch.nn.functional.pad(x, (0, pad_size), mode='constant', value=0)
            padded_signals.append(padded.unsqueeze(1))  # [B, 1, max_len]
        
        # 2. Объединение по каналам
        combined = torch.cat(padded_signals, dim=1)  # [B, M, max_len]
        
        # 3. Совместная обработка
        features = self.shared_conv(combined)
        
        # 4. Предсказание
        return self.final_fc(features)








# млп)
class LinearRegressor(nn.Module):
    def __init__(self, input_dims: list, num_targets: int):
        """
        :param input_dims: Список длин сигналов (например, [1000, 2000])
        :param num_targets: Количество целевых переменных (N)
        """
        super().__init__()
        self.num_experiments = len(input_dims)
        self.num_targets = num_targets
        self.input_dims = input_dims
        
        # Общая сумма размеров всех сигналов
        total_input_size = sum(input_dims)
        
        # Простая линейная регрессия
        self.linear_layers = nn.Sequential(
            nn.Linear(total_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_targets)
        )
        
        # Инициализация весов для стабильности
        self._initialize_weights()

    def _initialize_weights(self):
        """Инициализация весов для предотвращения NaN/Inf"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, *x_signals):
        """
        :param x_signals: M сигналов размером [B, L_i], где L_i может отличаться
        :return: Предсказания размером [B, num_targets]
        """
        # Проверка количества входных сигналов
        if len(x_signals) != self.num_experiments:
            raise ValueError(f"Expected {self.num_experiments} signals, got {len(x_signals)}")
        
        # Проверка размеров сигналов
        batch_size = x_signals[0].shape[0]
        
        # Собираем все сигналы в один вектор
        signal_vectors = []
        for i, signal in enumerate(x_signals):
            # Проверяем размерность
            if signal.shape[-1] != self.input_dims[i]:
                # Автоматически обрезаем или дополняем
                if signal.shape[-1] > self.input_dims[i]:
                    signal = signal[..., :self.input_dims[i]]
                else:
                    pad_size = self.input_dims[i] - signal.shape[-1]
                    signal = F.pad(signal, (0, pad_size), mode='constant', value=0)
            
            # Выравниваем сигнал
            flattened = signal.reshape(batch_size, -1)
            signal_vectors.append(flattened)
        
        # Конкатенация всех сигналов
        combined = torch.cat(signal_vectors, dim=1)  # [B, total_input_size]
        
        # Проходим через линейные слои
        return self.linear_layers(combined)

    def get_feature_importance(self, *x_signals):
        """
        Возвращает важность признаков для интерпретации модели
        """
        with torch.no_grad():
            # Получаем веса первого слоя
            first_layer = self.linear_layers[0]
            weights = first_layer.weight.data  # [128, total_input_size]
            
            # Абсолютные значения весов как мера важности
            importance = torch.abs(weights).mean(dim=0)  # [total_input_size]
            
            # Разделяем важность по сигналам
            importance_per_signal = []
            start_idx = 0
            for dim in self.input_dims:
                end_idx = start_idx + dim
                signal_importance = importance[start_idx:end_idx]
                importance_per_signal.append(signal_importance)
                start_idx = end_idx
            
            return importance_per_signal
        












from torch.nn import functional as F
class SVRegressorV0(nn.Module):
    """Максимально простая версия"""
    def __init__(self, input_dims, num_targets):
        super().__init__()
        self.total_size = sum(input_dims)
        
        # Просто линейная модель + RBF фичи
        self.linear = nn.Linear(self.total_size, 128)
        self.rbf_centers = nn.Parameter(torch.randn(10, 128) * 0.1)
        self.output_layer = nn.Linear(10, num_targets)
    
    def forward(self, *x_signals):
        # Объединяем
        combined = torch.cat([x.reshape(x.shape[0], -1) for x in x_signals], dim=1)
        
        # Линейное преобразование
        features = F.relu(self.linear(combined))
        
        # RBF активации
        distances = torch.cdist(features, self.rbf_centers)
        rbf_features = torch.exp(-0.1 * distances**2)
        
        return self.output_layer(rbf_features)

class SVRegressor(nn.Module):
    def __init__(self, input_dims: list, num_targets: int, num_support_vectors: int = 100, gamma: float = None):
        """
        Kernel Support Vector Regressor с RBF ядром
        
        :param input_dims: Список длин сигналов (например, [1000, 2000])
        :param num_targets: Количество целевых переменных (N)
        :param num_support_vectors: Количество опорных векторов
        :param gamma: Параметр RBF ядра (gamma = 1 / (2 * sigma^2))
        """
        super().__init__()
        self.num_experiments = len(input_dims)
        self.num_targets = num_targets
        self.input_dims = input_dims
        self.num_support_vectors = num_support_vectors
        self.gamma = gamma if gamma is not None else 0.1
        
        # Общая размерность входных данных
        self.total_input_size = sum(input_dims)
        
        # Инициализируем опорные векторы и их веса
        self.support_vectors = nn.Parameter(torch.randn(num_support_vectors, self.total_input_size))
        # self.alphas = nn.Parameter(torch.randn(num_support_vectors, num_targets) * 0.001)
        # self.bias = nn.Parameter(torch.zeros(num_targets))
        self.alphas = nn.Parameter(torch.zeros(num_support_vectors, num_targets))  # Начинаем с нулей
        self.bias = nn.Parameter(torch.zeros(num_targets))
        
        # Кэш для ускорения вычислений
        self.register_buffer('sv_norms', torch.zeros(num_support_vectors))
        self._update_sv_norms()
        
        # Для регуляризации
        # self.C = nn.Parameter(torch.tensor(1.0))
        self.C = 1.0
        
    def _rbf_kernel(self, X: torch.Tensor) -> torch.Tensor:
        """
        Вычисление RBF ядра: K(x, sv) = exp(-gamma * ||x - sv||^2)
        
        :param X: Входные данные [B, total_input_size]
        :return: Матрицу ядра [B, num_support_vectors]
        """        
        # # Вычисляем квадраты расстояний
        # X_norm = torch.sum(X**2, dim=1, keepdim=True)  # [B, 1]
        # # ||x - sv||^2 = ||x||^2 - 2*x·sv + ||sv||^2
        # distances = X_norm - 2 * X @ self.support_vectors.T + self.sv_norms.unsqueeze(0)  # [B, num_sv]
        # # Применяем RBF
        # K = torch.exp(-self.gamma * distances)

        # Численно стабильная версия RBF
        X_norm = (X ** 2).sum(dim=1, keepdim=True)
        sv_norm = self.sv_norms.unsqueeze(0)
        
        # Используем формулу с макс. стабильностью
        pairwise_dist = X_norm + sv_norm - 2 * torch.mm(X, self.support_vectors.t())
        pairwise_dist = torch.clamp(pairwise_dist, min=0)  # Убираем отрицательные из-за ошибок округления
        
        K = torch.exp(-self.gamma * pairwise_dist)        
        return K
    
    def _update_sv_norms(self):
        """Обновление квадратов норм опорных векторов"""
        with torch.no_grad():
            self.sv_norms.copy_(torch.sum(self.support_vectors**2, dim=1))
    
    def _combine_signals(self, *x_signals: torch.Tensor) -> torch.Tensor:
        """
        Объединение сигналов в один вектор
        
        :param x_signals: M сигналов размером [B, L_i]
        :return: Объединенный вектор [B, total_input_size]
        """
        batch_size = x_signals[0].shape[0]
        signal_vectors = []
        
        for i, signal in enumerate(x_signals):
            # Проверяем и корректируем размерность
            if signal.shape[-1] != self.input_dims[i]:
                if signal.shape[-1] > self.input_dims[i]:
                    signal = signal[..., :self.input_dims[i]]
                else:
                    pad_size = self.input_dims[i] - signal.shape[-1]
                    signal = F.pad(signal, (0, pad_size), mode='constant', value=0)
            
            # Выравниваем
            flattened = signal.reshape(batch_size, -1)
            signal_vectors.append(flattened)
        
        # Конкатенация
        return torch.cat(signal_vectors, dim=1)
    
    def forward(self, *x_signals: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход SVR
        
        :param x_signals: M сигналов разной длины
        :return: Предсказания [B, num_targets]
        """
        # 1. Объединяем сигналы
        X = self._combine_signals(*x_signals)  # [B, total_input_size]
        
        # X = F.normalize(X, dim=1, eps=1e-8)
        # 2. Вычисляем RBF ядро
        K = self._rbf_kernel(X)  # [B, num_sv]
        
        # 3. SVR предсказание: f(x) = Σ α_i * K(x, sv_i) + b
        predictions = K @ self.alphas + self.bias.unsqueeze(0)  # [B, num_targets]
        
        return predictions
    
    def compute_svr_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                        epsilon: float = 0.1) -> torch.Tensor:
        """
        Epsilon-insensitive loss для SVR + регуляризация
        
        :param predictions: Предсказания модели [B, num_targets]
        :param targets: Истинные значения [B, num_targets]
        :param epsilon: Параметр эпсилон-нечувствительности
        :return: Полный loss
        """
        # Epsilon-insensitive loss
        errors = predictions - targets


        # epsilon_loss = torch.max(torch.abs(errors) - epsilon, torch.zeros_like(errors))
        abs_errors = torch.abs(errors)
        epsilon_loss = F.relu(abs_errors - epsilon)


        data_loss = torch.mean(epsilon_loss)
        
        # Регуляризация (норма весов)
        reg_loss = torch.mean(self.alphas**2)
        
        # Полный loss с параметром C
        # total_loss = data_loss + (1.0 / self.C) * reg_loss
        total_loss = self.C * data_loss + 0.01 * reg_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return data_loss #FALLBACK
        
        return total_loss
    
    def get_support_vectors(self) -> torch.Tensor:
        """Возвращает текущие опорные векторы"""
        return self.support_vectors.detach()
    
    def get_feature_importance(self, *x_signals: torch.Tensor) -> list:
        """
        Оценка важности признаков через градиенты ядра
        
        :param x_signals: Входные сигналы
        :return: Список важностей для каждого сигнала
        """
        X = self._combine_signals(*x_signals)
        X.requires_grad_(True)
        
        # Вычисляем градиенты ядра по входам
        K = self._rbf_kernel(X)
        # Важность = среднее абсолютное значение градиента
        importance = torch.autograd.grad(K.sum(), X, retain_graph=True)[0]
        importance = torch.abs(importance).mean(dim=0)
        
        # Разделяем по сигналам
        importance_per_signal = []
        start_idx = 0
        for dim in self.input_dims:
            end_idx = start_idx + dim
            signal_importance = importance[start_idx:end_idx]
            importance_per_signal.append(signal_importance)
            start_idx = end_idx
        
        return importance_per_signal
    
    def update_support_vectors(self, new_vectors: torch.Tensor):
        """
        Обновление опорных векторов (можно использовать после обучения)
        
        :param new_vectors: Новые опорные векторы [num_sv, total_input_size]
        """
        with torch.no_grad():
            self.support_vectors.copy_(new_vectors)
            self._update_sv_norms()

# НЕ ОЧЕНЬ СРАБОТАЛО
    def initialize_from_data(self, dataloader, device):
        """Инициализация параметров из данных с адаптацией под размер выборки"""
        print("Initializing model from data...")
        
        # Собираем данные
        all_X = []
        all_y = []
        with torch.no_grad():
            for batch in dataloader:
                *x_batch, y_batch = batch
                x_batch = [x.to(device) for x in x_batch]
                X = self._combine_signals(*x_batch)
                all_X.append(X)
                all_y.append(y_batch.to(device))
                if len(all_X) * batch[0].shape[0] > 5000:
                    break
        
        all_X = torch.cat(all_X, dim=0)
        all_y = torch.cat(all_y, dim=0)
        
        print(f"Collected {len(all_X)} samples for initialization")
        
        # Адаптивное количество опорных векторов
        # Для маленькой выборки используем меньше SV
        if len(all_X) < 50:
            # Для очень маленькой выборки: SV = min(30, n_samples-1)
            adapted_sv_count = min(30, len(all_X) - 1)
            # Увеличиваем регуляризацию
            self.C = 10.0  # Сильнее штрафуем ошибки
        elif len(all_X) < 100:
            adapted_sv_count = min(50, len(all_X) - 1)
            self.C = 5.0
        else:
            adapted_sv_count = min(self.num_support_vectors, len(all_X))
            self.C = 1.0
        
        if adapted_sv_count < self.num_support_vectors:
            print(f"Adapting num_support_vectors from {self.num_support_vectors} to {adapted_sv_count} (dataset size: {len(all_X)})")
            self.num_support_vectors = adapted_sv_count
            # Пересоздаем параметры
            self.support_vectors = nn.Parameter(torch.randn(adapted_sv_count, self.total_input_size))
            self.alphas = nn.Parameter(torch.zeros(adapted_sv_count, self.num_targets))
            self.register_buffer('sv_norms', torch.zeros(adapted_sv_count))
            self._update_sv_norms()
        
        # Инициализация support_vectors из данных
        indices = torch.randperm(len(all_X))[:self.num_support_vectors]
        self.support_vectors.data = all_X[indices].clone()
        self._update_sv_norms()
        
        # Адаптивная gamma для маленькой выборки
        if self.gamma is None or self.gamma == 0.1:
            with torch.no_grad():
                dists = torch.cdist(self.support_vectors, self.support_vectors)
                non_zero_dists = dists[dists > 0]
                if len(non_zero_dists) > 0:
                    median_dist = torch.median(non_zero_dists).item()
                    
                    # Для маленькой выборки увеличиваем gamma (более локальное ядро)
                    if len(all_X) < 50:
                        # Более высокая gamma для лучшего разделения
                        self.gamma = 2.0 / (median_dist ** 2)
                    else:
                        self.gamma = 1.0 / (2 * (median_dist ** 2))
                    
                    self.gamma = max(0.05, min(20.0, self.gamma))
                    print(f"Gamma initialized to: {self.gamma:.4f} (median distance: {median_dist:.4f})")
                else:
                    self.gamma = 2.0 if len(all_X) < 50 else 1.0
                    print(f"Warning: Could not estimate gamma, using default: {self.gamma}")
        
        # Инициализация bias
        self.bias.data = all_y.mean(dim=0)
        
        # Для маленькой выборки используем меньшие начальные alpha
        if len(all_X) < 50:
            self.alphas.data = torch.randn_like(self.alphas) * 0.001
        else:
            self.alphas.data = torch.randn_like(self.alphas) * 0.01
        
        print(f"Initialization complete. Support vectors: {self.num_support_vectors}, Gamma: {self.gamma:.4f}, C: {self.C}")
# БЭКАП
    # def initialize_from_data(self, dataloader, device):
    #     """Инициализация параметров из данных"""
    #     print("Initializing model from data...")
        
    #     # Собираем данные
    #     all_X = []
    #     all_y = []
    #     with torch.no_grad():
    #         for batch in dataloader:
    #             *x_batch, y_batch = batch
    #             x_batch = [x.to(device) for x in x_batch]
    #             X = self._combine_signals(*x_batch)
    #             all_X.append(X)
    #             all_y.append(y_batch.to(device))
    #             if len(all_X) * batch[0].shape[0] > 5000:  # Ограничиваем размер
    #                 break
        
    #     all_X = torch.cat(all_X, dim=0)
    #     all_y = torch.cat(all_y, dim=0)
        
    #     print(f"Collected {len(all_X)} samples for initialization")
        
    #     # Корректируем количество опорных векторов, если нужно
    #     actual_sv_count = min(self.num_support_vectors, len(all_X))
    #     if actual_sv_count < self.num_support_vectors:
    #         print(f"Warning: Reducing num_support_vectors from {self.num_support_vectors} to {actual_sv_count} due to insufficient data")
    #         self.num_support_vectors = actual_sv_count
    #         # Пересоздаем параметры с новым размером
    #         self.support_vectors = nn.Parameter(torch.randn(actual_sv_count, self.total_input_size))
    #         self.alphas = nn.Parameter(torch.zeros(actual_sv_count, self.num_targets))
    #         self.register_buffer('sv_norms', torch.zeros(actual_sv_count))
    #         self._update_sv_norms()
        
    #     # Инициализация support_vectors из данных
    #     indices = torch.randperm(len(all_X))[:self.num_support_vectors]
    #     self.support_vectors.data = all_X[indices].clone()
    #     self._update_sv_norms()
        
    #     # Инициализация gamma на основе расстояний
    #     if self.gamma is None or self.gamma == 0.1:  # Если gamma не была задана или используется значение по умолчанию
    #         # Вычисляем расстояния между опорными векторами
    #         with torch.no_grad():
    #             dists = torch.cdist(self.support_vectors, self.support_vectors)
    #             # Берем ненулевые расстояния
    #             non_zero_dists = dists[dists > 0]
    #             if len(non_zero_dists) > 0:
    #                 median_dist = torch.median(non_zero_dists).item()
    #                 # gamma = 1 / (2 * sigma^2), где sigma - масштаб
    #                 # Используем медианное расстояние как оценку sigma
    #                 self.gamma = 1.0 / (2 * (median_dist ** 2))
    #                 # Ограничиваем gamma разумными пределами
    #                 self.gamma = max(0.01, min(10.0, self.gamma))
    #                 print(f"Gamma initialized to: {self.gamma:.4f} (median distance: {median_dist:.4f})")
    #             else:
    #                 self.gamma = 1.0
    #                 print(f"Warning: Could not estimate gamma, using default: {self.gamma}")
        
    #     # Инициализация bias как среднее целевых значений
    #     self.bias.data = all_y.mean(dim=0)
        
    #     # Инициализация alphas (маленькие случайные значения)
    #     self.alphas.data = torch.randn_like(self.alphas) * 0.01
        
    #     print(f"Initialization complete. Support vectors: {self.num_support_vectors}, Gamma: {self.gamma:.4f}")








# тру регрессия
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TrueLR(nn.Module):
    def __init__(self, input_dims: list, num_targets: int):
        """
        :param input_dims: Список длин сигналов (например, [1000, 2000])
        :param num_targets: Количество целевых переменных (N)
        """
        super().__init__()
        self.num_experiments = len(input_dims)
        self.num_targets = num_targets
        self.input_dims = input_dims
        
        # общая сумма размеров всех сигналов
        total_input_size = sum(input_dims)
        
        # Единственный линейный слой — это и есть классическая линейная регрессия
        # Веса размера [num_targets, total_input_size]
        # Смещение (bias) размера [num_targets]
        self.linear = nn.Linear(total_input_size, num_targets)
        
        self._initialize_weights()

    def _initialize_weights(self):
        # Стандартная инициализация для линейной регрессии
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Инициализация весов по умолчанию из PyTorch (Kaiming Uniform)
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, *x_signals):
        """
        :param x_signals: M сигналов размером [B, L_i], где L_i - длина i-го сигнала
        :return: Предсказания размером [B, num_targets]
        """
        if len(x_signals) != self.num_experiments:
            raise ValueError(f"Expected {self.num_experiments} signals, got {len(x_signals)}")
        
        batch_size = x_signals[0].shape[0]
        
        # все сигналы в один вектор
        signal_vectors = []
        for i, signal in enumerate(x_signals):
            if signal.shape[-1] != self.input_dims[i]:
                if signal.shape[-1] > self.input_dims[i]:
                    signal = signal[..., :self.input_dims[i]]
                else:
                    pad_size = self.input_dims[i] - signal.shape[-1]
                    signal = F.pad(signal, (0, pad_size), mode='constant', value=0)

            signal_vectors.append(signal)
        
        combined = torch.cat(signal_vectors, dim=1)

        return self.linear(combined) # [batch_size, num_targets]

    def get_feature_importance(self, *x_signals):
        """
        Возвращает веса линейной модели для каждого входного отсчёта.
        Для линейной регрессии важность признака прямо пропорциональна |веса|
        """
        # weights имеет размер [num_targets, total_input_size]
        # Усредняем по целевым переменным, если их несколько
        importance = torch.abs(self.linear.weight.data).mean(dim=0)  # [total_input_size]
        
        # Разбиваем важность по исходным сигналам
        importance_per_signal = []
        start_idx = 0
        for dim in self.input_dims:
            end_idx = start_idx + dim
            signal_importance = importance[start_idx:end_idx]
            importance_per_signal.append(signal_importance)
            start_idx = end_idx
        
        return importance_per_signal