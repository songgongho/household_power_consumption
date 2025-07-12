# -*- coding: utf-8 -*-
"""
과제명: 다양한 딥러닝 모델을 활용한 전력 수요 예측 모델 구현
- Individual Household Electric Power Consumption 데이터셋 활용
- Global_active_power 컬럼 기반 일별 평균 전력 예측
- 80% 학습 / 20% 테스트, 시계열 모델 4종(LSTM, GRU, BiLSTM, CNN-LSTM)
- 성능 평가: MAE, RMSE
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 불러오기 및 전처리
def load_and_preprocess(file_path='household_power_consumption.txt'):
    df = pd.read_csv(
        file_path,
        sep=';',
        parse_dates={'Datetime': ['Date', 'Time']},
        dayfirst=True,
        na_values='?',
        low_memory=False
    )
    print(f"원본 데이터 크기: {df.shape}")
    print(df.head())

    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df.dropna(inplace=True)
    df.set_index('Datetime', inplace=True)
    daily_power = df['Global_active_power'].resample('D').mean()
    daily_power.ffill(inplace=True)  # 결측치 전방 채움
    return daily_power

# 2. 시퀀스 생성
def create_sequences(series, seq_len=30):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaler

# 3. 모델 정의
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilstm = nn.LSTM(1, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.fc(out[:, -1])

class CNNLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(16, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(torch.relu(self.conv1d(x)))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# 4. 모델 학습 및 평가
def train_and_evaluate(model, train_loader, test_loader, scaler, epochs=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            pred = model(batch_x)
            predictions.append(pred.numpy())
            actuals.append(batch_y.numpy())

    pred_all = np.vstack(predictions)
    actual_all = np.vstack(actuals)
    pred_inv = scaler.inverse_transform(pred_all)
    actual_inv = scaler.inverse_transform(actual_all)
    mae = mean_absolute_error(actual_inv, pred_inv)
    rmse = np.sqrt(mean_squared_error(actual_inv, pred_inv))

    return mae, rmse, actual_inv.flatten(), pred_inv.flatten()

# 5. 전체 실행
def main():
    file_path = "household_power_consumption.txt"
    data = load_and_preprocess(file_path)
    X, y, scaler = create_sequences(data)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32)

    models = {
        'LSTM': LSTMModel(),
        'GRU': GRUModel(),
        'BiLSTM': BiLSTMModel(),
        'CNN-LSTM': CNNLSTMModel()
    }

    results = {}
    for name, model in models.items():
        print(f"\n🔧 {name} 모델 학습 시작...")
        start = time.time()
        mae, rmse, actual, predicted = train_and_evaluate(model, train_loader, test_loader, scaler)
        elapsed = time.time() - start
        results[name] = (mae, rmse, elapsed)

        plt.figure(figsize=(10, 4))
        plt.plot(actual, label='실제값')
        plt.plot(predicted, label='예측값')
        plt.title(f"{name} 예측 결과")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("\n📊 모델별 성능 요약:")
    best = min(results.items(), key=lambda x: x[1][1])[0]
    for name, (mae, rmse, t) in results.items():
        mark = "✅" if name == best else ""
        print(f"{name:10} → MAE: {mae:.3f}, RMSE: {rmse:.3f}, Time: {t:.2f}s {mark}")

if __name__ == "__main__":
    main()
