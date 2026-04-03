import time

import pandas as pd
import re

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
import torch.nn.functional
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub(r'[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


def getData():
    df = pd.read_csv('movie_data.csv')

    df['review'] = df['review'].apply(preprocessor)

    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.3, random_state=1)

    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=True,
                            preprocessor=None,
                            ngram_range=(1, 2),
                            max_features=10000)

    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    return train_dataset, test_dataset


class BasicNN(nn.Module):
    def __init__(self, input_dim):
        super(BasicNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 64)
        self.bn2 = nn.BatchNorm1d(64)
        # self.fc3 = nn.Linear(128, 64)
        # self.bn3 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn2(self.fc2(x)))
        # x = torch.nn.functional.relu(self.bn3(self.fc3(x)))
        x = torch.sigmoid(self.out(x))
        return x


def test_logistic_regression(train_data, test_data):

    X_train = train_data.tensors[0].numpy()
    y_train = train_data.tensors[1].numpy().ravel()
    X_test = test_data.tensors[0].numpy()
    y_test = test_data.tensors[1].numpy().ravel()


    lr_model = LogisticRegression(solver='liblinear',random_state=1)


    start_time = time.time()
    lr_model.fit(X_train, y_train)
    end_time = time.time()

    y_pred = lr_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    y_pred = lr_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred)

    print(f'Training Accuracy: {test_accuracy* 100:.2f}%')
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Logistic Regression Training Time: {end_time - start_time:.2f}s")



def train_validate_model(model, train_data, test_data):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rate = 0.00001
    num_epochs = 5
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    total_time = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted.view(-1) == labels.view(-1)).sum().item()
            total += labels.size(0)

            optimizer.zero_grad()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        end_time = time.time()
        total_time += end_time - start_time
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted.view(-1) == labels.view(-1)).sum().item()
                total += labels.size(0)

        test_loss /= len(test_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(test_loss)
        val_accuracies.append(val_accuracy)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Val Loss: {test_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    print(f'Total Training Time: {total_time:.2f}s')
    return train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == "__main__":
    print(device)
    train_data, test_data = getData()
    model = BasicNN(train_data.tensors[0].shape[1]).to(device)
    print('Testing for BasicNN\n')
    train_validate_model(model, train_data, test_data)

    print('\nTesting for Logistic Regression\n')
    test_logistic_regression(train_data, test_data)


