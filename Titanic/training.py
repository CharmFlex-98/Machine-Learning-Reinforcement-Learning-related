import torch
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import math
from sklearn import preprocessing

train = False
data_path = r'C:\Users\ASUS\PycharmProjects\Pytorch\Titanic\train.csv'
test_path = r'C:\Users\ASUS\PycharmProjects\Pytorch\Titanic\test.csv'
learning_rate = 0.001
epoch = 100
device = torch.device('cuda')
weight_path = 'best_weight.pt'

x=0
def det_var(data, model_data=None):
    if train:
        data = data.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket'])
    else:
        data = data.drop(columns=['PassengerId', 'Name', 'Ticket'])
        model_data = model_data.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket'])

    data_numeric = data.select_dtypes(exclude=['object'])
    data_categorial = data.select_dtypes(include=['object'])

    tensor = []
    model_tensor=[]
    ohe = preprocessing.OneHotEncoder(handle_unknown='ignore')

    for feature in data:
        if feature in data_numeric:
            if data[feature].isnull().sum() > 0:
                feature_mean = data[feature].mean()
                data[feature] = data[feature].fillna(feature_mean)
            if not train:
                if model_data[feature].isnull().sum() > 0:
                    feature_mean = model_data[feature].mean()
                    model_data[feature] = model_data[feature].fillna(feature_mean)
                model_tensor.append(torch.tensor(model_data[feature]).float().unsqueeze(1))

            tensor.append(torch.tensor(data[feature]).float().unsqueeze(1))

        elif feature in data_categorial:
            if data[feature].isnull().sum() > 0:
                data[feature] = data[feature].fillna('Nan')

            if feature == 'Cabin':
                for element in data[feature]:
                    data[feature] = data[feature].replace({element: element[0]})

            if not train:
                if model_data[feature].isnull().sum() > 0:
                    model_data[feature] = model_data[feature].fillna('Nan')
                if feature == 'Cabin':
                    for element in model_data[feature]:
                        model_data[feature] = model_data[feature].replace({element: element[0]})

            if train:
                ohe_label = np.array(data[feature]).reshape(-1, 1)
                ohe.fit(ohe_label)
            else:
                model_ohe = np.array(model_data[feature]).reshape(-1, 1)
                ohe.fit(model_ohe)
                ohe_label = np.array(data[feature]).reshape(-1, 1)

            ohe_label = ohe.transform(ohe_label).toarray()
            if not train:
                model_ohe=ohe.transform(model_ohe).toarray()
                model_tensor.append(torch.tensor(model_ohe).float())
            tensor.append(torch.tensor(ohe_label).float())

        else:
            print('Error: Unknown feature! ')

    tensor = torch.cat(tensor, 1)
    if not train:
        model_tensor=torch.cat(model_tensor, 1)

    if train:
        min = tensor.min(0)[0]
        max = tensor.max(0)[0]
    else:
        min = model_tensor.min(0)[0]
        max = model_tensor.max(0)[0]

    tensor=(tensor-min)/(max-min)

    print(tensor[0])
    return tensor


def count_accuracy(prediction, label):
    num_correct = torch.argmax(prediction, 1).eq(label).sum().item()
    total_num = len(label)

    accuracy = float((num_correct / total_num) * 100)

    return accuracy


def write_result(prediction, passenger_id):
    output = torch.argmax(prediction, 1).item()
    passenger_id = passenger_id.item()

    return '{},{}{}'.format(passenger_id, output, '\n')


class titanic_Dataset(Dataset):
    def __init__(self, train_data, test_data=None):
        if train:
            self.data = pd.read_csv(train_data)
            self.variable = det_var(self.data)
        else:
            self.data = pd.read_csv(test_data)
            self.model_data = pd.read_csv(train_data)
            self.variable = det_var(self.data, self.model_data)

    def __getitem__(self, index):
        variable = self.variable[index]
        if train:
            label = self.data['Survived'][index]
            return variable, label
        else:
            PI = self.data['PassengerId'][index]
            return variable, PI

    def __len__(self):
        return len(self.data)


class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        self.linear = nn.Linear(20, 50)
        self.batch_norm = nn.BatchNorm1d(50)
        self.linear2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.batch_norm(self.linear(x)))
        output = self.linear2(x)

        return output


# det_var(pd.read_csv(data_path))

if train:
    dataset = titanic_Dataset(data_path)
else:
    dataset = titanic_Dataset(data_path, test_path)

if train:
    data_train, data_valid = random_split(dataset, [713, 178])
    train_loader = DataLoader(data_train, batch_size=32, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=32, shuffle=True, drop_last=True)
else:
    test_loader = DataLoader(dataset, batch_size=1)

model = my_model()
model.to(device=device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

if train:
    valid_loss_list = []
    accuracy_list = []
    best_accuracy = ''
    for _epoch in range(epoch):
        valid_loss = 0
        accuracy = 0
        model.train()
        for batch in train_loader:
            variable, label = batch
            variable = variable.to(device=device)
            label = label.to(device=device)

            prediction = model(variable)

            loss = loss_function(prediction, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        for batch in valid_loader:
            variable, label = batch
            variable = variable.to(device=device)
            label = label.to(device=device)

            with torch.no_grad():
                prediction = model(variable)
                loss = loss_function(prediction, label)

                valid_loss += loss.item()
                accuracy += count_accuracy(F.log_softmax(prediction, 1), label)

        valid_loss_list.append(valid_loss)
        accuracy = round(accuracy / len(valid_loader), 1)
        accuracy_list.append(accuracy)
        print('loss:', valid_loss_list)
        print('accuracy:', accuracy_list)

        if best_accuracy:
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                print('highest accuracy at epoch {}!'.format(_epoch))
                torch.save(model.state_dict(), weight_path)
        else:
            best_accuracy = accuracy

else:
    submission_file = open('submission.csv', 'w')
    submission_file.writelines('PassengerId,Survived\n')
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    for batch in test_loader:
        variable, PassengerId = batch
        variable = variable.to(device)
        PassengerId = PassengerId.to(device)

        with torch.no_grad():
            output = model(variable)

        submission_file.writelines(write_result(output, PassengerId))

    submission_file.close()
