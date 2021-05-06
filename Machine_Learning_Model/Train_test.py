import torch
import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt


train = True
train_path = r'C:\Users\ASUS\PycharmProjects\Pytorch\HousePrice\train.csv'
test_path = r'C:\Users\ASUS\PycharmProjects\Pytorch\HousePrice\test.csv'
weight_path = r'C:\Users\ASUS\PycharmProjects\Pytorch\HousePrice\best_weight.pt'
write_path = r'C:\Users\ASUS\PycharmProjects\Pytorch\HousePrice\submission.csv'
figure_path = r'C:\Users\ASUS\PycharmProjects\Pytorch\HousePrice\loss_graph.jpg'
batch_size = 128
learning_rate = 0.01
epoch = 200
input_layer = 576
output_layer = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ohe = preprocessing.OneHotEncoder(handle_unknown='ignore')


def preprocess_data(data, model_data=None):
    if train:
        data = data.drop(columns=['Id', 'SalePrice'])
        categorize = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
        for feature in categorize:
            data[feature] = data[feature].astype('str')

    else:
        data = data.drop(columns=['Id'])
        categorize = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
        for feature in categorize:
            data[feature] = data[feature].astype('str')

        model_data = model_data.drop(columns=['Id', 'SalePrice'])
        categorize = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
        for feature in categorize:
            model_data[feature] = model_data[feature].astype('str')

    numeric = data.select_dtypes(exclude=['object'])
    categorial = data.select_dtypes(include=['object'])

    tensor = []
    model_tensor = []

    for feature in data:
        if feature in numeric:
            if data[feature].isnull().sum() > 0:
                feature_mean = data[feature].mean()
                data[feature] = data[feature].fillna(feature_mean)
            if not train:
                if model_data[feature].isnull().sum() > 0:
                    feature_mean = model_data[feature].mean()
                    model_data[feature] = model_data[feature].fillna(feature_mean)
                model_tensor.append(torch.tensor(model_data[feature]).float().unsqueeze(1))
            tensor.append(torch.tensor(data[feature]).float().unsqueeze(1))

        elif feature in categorial:
            if data[feature].isnull().sum() > 0:
                data[feature] = data[feature].fillna('NaN')

            ohe_label = np.array(data[feature]).reshape(-1, 1)

            if not train:
                if model_data[feature].isnull().sum() > 0:
                    model_data[feature] = model_data[feature].fillna('NaN')
                model_ohe_label = np.array(model_data[feature]).reshape(-1, 1)

            if train:
                ohe.fit(ohe_label)
            else:
                ohe.fit(model_ohe_label)
                model_ohe_label = ohe.transform(model_ohe_label).toarray()
                model_tensor.append(torch.tensor(model_ohe_label).float())

            ohe_label = ohe.transform(ohe_label).toarray()
            tensor.append(torch.tensor(ohe_label).float())

        else:
            print('Unknown feature!')

    tensor = torch.cat(tensor, 1)

    if not train:
        model_tensor = torch.cat(model_tensor, 1)
    if train:
        min_ = tensor.min(0)[0]
        max_ = tensor.max(0)[0]
    else:
        min_ = model_tensor.min(0)[0]
        max_ = model_tensor.max(0)[0]

    tensor = (tensor - min_) / (max_ - min_)

    print(tensor.shape)

    return tensor


def write_result(_prediction, _id, _file):
    _prediction = _prediction.item()
    _id = _id.item()
    _file.writelines('{},{}{}'.format(_id, _prediction, '\n'))


def plot_chart(y1, y2, x, total_epoch):
    x = [value + 1 for value in range(x)]
    plt.cla()
    plt.title('TRAIN-VALID LOSS CURVE')
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.plot(x, y1, label='train loss')
    plt.plot(x, y2, label='valid loss')
    plt.xlim([0, total_epoch])
    plt.ylim(bottom=0)  # put xylim after plot! Or else it would not rescale to the upper limit!
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.style.use('ggplot')
    plt.pause(0.001)


class my_dataset(Dataset):
    def __init__(self, train_set, test_set=None):
        if train:
            self.data = pd.read_csv(train_set)
            self.variable = preprocess_data(self.data)
        else:
            self.data = pd.read_csv(test_set)
            self.model_data = pd.read_csv(train_set)
            self.variable = preprocess_data(self.data, self.model_data)

    def __getitem__(self, index):
        variable = self.variable[index]
        if train:
            label = self.data['SalePrice'][index]

            return variable, label
        else:
            ID = self.data['Id'][index]

            return variable, ID

    def __len__(self):
        return len(self.data)


# preprocess_data(pd.read_csv(r'C:\Users\ASUS\PycharmProjects\Pytorch\Loan\test_lAUu6dG(1).csv'),
#                 pd.read_csv(r'C:\Users\ASUS\PycharmProjects\Pytorch\Loan\train_ctrUa4K(1).csv'))

class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        self.linear1 = nn.Linear(input_layer, input_layer * 5)
        self.BatchNorm1 = nn.BatchNorm1d(self.linear1.out_features)
        self.linear2 = nn.Linear(self.linear1.out_features, int(self.linear1.out_features / 10))
        self.BatchNorm2 = nn.BatchNorm1d(self.linear2.out_features)
        self.linear3 = nn.Linear(self.linear2.out_features, output_layer)

    def forward(self, x):
        x = f.relu(self.BatchNorm1(self.linear1(x)))
        x = f.relu(self.BatchNorm2(self.linear2(x)))
        output = self.linear3(x)

        return output


if train:
    dataset = my_dataset(train_path)
else:
    dataset = my_dataset(train_path, test_path)
model = my_model()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=int(epoch / 10), verbose=True)
loss_function = nn.MSELoss()
plt.ion()

if train:
    num_train_set = int(len(dataset.data) * 90 / 100)
    num_valid_set = len(dataset.data) - num_train_set

    train_set, valid_set = random_split(dataset, [num_train_set, num_valid_set])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)

    train_loss_history = []
    valid_loss_history = []
    least_loss = ''
    for _epoch in range(epoch):
        train_loss = 0
        valid_loss = 0
        for loader in [train_loader, valid_loader]:
            model.train() if loader == train_loader else model.eval()
            with torch.set_grad_enabled(loader == train_loader):
                for batch in loader:
                    variable, label = batch
                    variable = variable.to(device)
                    label = label.float().unsqueeze(1).to(device)
                    prediction = model(variable)

                    loss = loss_function(prediction, label)

                    if loader == train_loader:
                        loss.backward()

                        optimizer.step()
                        optimizer.zero_grad()
                        train_loss += loss.item()
                    else:
                        valid_loss += loss.item()

        valid_loss = float(valid_loss / len(valid_set))
        train_loss = float(train_loss / len(train_set))

        valid_loss_history.append(valid_loss)
        train_loss_history.append(train_loss)

        if least_loss:
            if least_loss > valid_loss:
                least_loss = valid_loss
                print('least_loss at epoch {}\nvalid_loss: {}'.format(_epoch + 1, valid_loss))
                torch.save(model.state_dict(), weight_path)
        else:
            least_loss = valid_loss
            print('least_loss at epoch {}\nvalid_loss: {}'.format(_epoch + 1, valid_loss))
            torch.save(model.state_dict(), weight_path)

        scheduler.step(valid_loss)

        plot_chart(train_loss_history, valid_loss_history, _epoch + 1, epoch)
    plt.savefig(figure_path)

else:
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    test_loader = DataLoader(dataset, batch_size=1)
    with open(write_path, 'w') as file:
        file.writelines('{},{}{}'.format('Id', 'SalePrice', '\n'))
        with torch.no_grad():
            for batch in test_loader:
                variable, ID = batch
                variable = variable.to(device)
                prediction = model(variable)
                write_result(prediction, ID, file)
