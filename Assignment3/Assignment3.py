import pandas as pd
import re

import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
import torch.nn.functional
from torch.utils.data import TensorDataset, DataLoader

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
    def __init__(self,input_dim):
        super(BasicNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn2(self.fc2(x)))
        x = torch.nn.functional.relu(self.bn3(self.fc3(x)))
        x = torch.sigmoid(self.out(x))
        return x



def testNN(model,data_set):
    learning_rate = 0.000045
    epochs = 5

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
    L = torch.nn.BCELoss()

    train_data = DataLoader(data_set, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for(x,y) in train_data:
            x, y = x.to(device), y.to(device)

            output = model.forward(x)
            loss = L(output, y)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_data)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
    return model


def evaluate(model, test_dataset):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=100)
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            predicted = torch.round(outputs)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f'Accuracy on test data: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    print(device)
    train_data, test_data = getData()
    model = BasicNN(train_data.tensors[0].shape[1]).to(device)
    trained_model = testNN(model, train_data)

    evaluate(trained_model, test_data)

