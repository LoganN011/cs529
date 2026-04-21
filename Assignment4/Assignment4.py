import math
import time

import numpy as np
import torch
import torch.optim as optim
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getData(tickers, M=50, N=1, train_split=0.8):
    all_data = {}

    for ticker in tickers:
        df = yf.download(ticker, start='2018-02-01', end='2026-01-31', progress=False)
        if df.empty: continue

        data = df['Close'].values.reshape(-1, 1)

        split_idx = int(len(data) * train_split)
        train_raw = data[:split_idx]
        test_raw = data[split_idx:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_raw)

        train_scaled = scaler.transform(train_raw)
        test_scaled = scaler.transform(test_raw)

        full_scaled = np.vstack((train_scaled, test_scaled))

        X, y = [], []
        for i in range(len(full_scaled) - M - N + 1):
            X.append(full_scaled[i: i + M])
            y.append(full_scaled[i + M: i + M + N])

        X, y = np.array(X), np.array(y)

        window_split = split_idx - M

        X_train = torch.tensor(X[:window_split], dtype=torch.float32)
        y_train = torch.tensor(y[:window_split], dtype=torch.float32)
        X_test = torch.tensor(X[window_split:], dtype=torch.float32)
        y_test = torch.tensor(y[window_split:], dtype=torch.float32)

        all_data[ticker] = {
            'train': (X_train, y_train),
            'test': (X_test, y_test),
            'scaler': scaler
        }
    return all_data


class BasicRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(BasicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, hn = self.rnn(x, h0)

        last_hidden = hn[-1]

        out = self.bn(last_hidden)
        return self.fc(out)


class GRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, hn = self.gru(x, h0)

        last_hidden = hn[-1]

        out = self.bn(last_hidden)
        return self.fc(out)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        last_hidden = hn[-1]

        out = self.bn(last_hidden)
        return self.fc(out)


def test_model(model, train_loader, test_loader, scaler, epochs=50, lr=0.001):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    start_time = time.time()

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.squeeze(-1))
            loss.backward()
            optimizer.step()
        scheduler.step()

    total_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        test_X, test_y = test_loader.dataset.tensors
        test_X, test_y = test_X.to(device), test_y.to(device)
        test_preds = model(test_X)

        test_y_np = test_y.cpu().numpy().reshape(-1, 1)
        test_preds_np = test_preds.cpu().numpy().reshape(-1, 1)

        actual_prices = scaler.inverse_transform(test_y_np).reshape(-1, output_size)
        predicted_prices = scaler.inverse_transform(test_preds_np).reshape(-1, output_size)

        mae = np.mean(np.abs(actual_prices - predicted_prices))

        mse = np.mean((actual_prices - predicted_prices) ** 2)

        rmse = math.sqrt(mse)

        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices))
        test_acc = max(0, (1 - mape) * 100)

    print(f"Time Cost of Learning: {total_time:.2f} seconds")
    print(f"Mean Absolute Error (MAE):    {mae:.4f}")
    print(f"Mean Squared Error (MSE):     {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")
    print(f"Estimated Test Accuracy:      {test_acc:.2f}%")
    print()


if __name__ == '__main__':
    print(f"Using device: {device}")
    tickers = ['AAPL', 'GOOG', 'AMD', 'NVDA']

    output_size = 1
    all_data = getData(tickers, M=50, N=output_size)

    print('Basic RNN\n')
    for ticker in tickers:
        print(ticker)
        data = all_data[ticker]
        train_ds = TensorDataset(data['train'][0], data['train'][1])
        test_ds = TensorDataset(data['test'][0], data['test'][1])

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        modelRNN = BasicRNN(output_size=output_size)
        test_model(modelRNN, train_loader, test_loader, data['scaler'])

    print('\nGRU\n')
    for ticker in tickers:
        print(ticker)
        data = all_data[ticker]
        train_ds = TensorDataset(data['train'][0], data['train'][1])
        test_ds = TensorDataset(data['test'][0], data['test'][1])

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        modelGRU = GRU(output_size=output_size)
        test_model(modelGRU, train_loader, test_loader, data['scaler'])

    print('\nLSTM\n')
    for ticker in tickers:
        print(ticker)
        data = all_data[ticker]
        train_ds = TensorDataset(data['train'][0], data['train'][1])
        test_ds = TensorDataset(data['test'][0], data['test'][1])

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        modelLSTM = LSTM(output_size=output_size)
        test_model(modelLSTM, train_loader, test_loader, data['scaler'])
