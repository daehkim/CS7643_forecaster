import numpy as np
import pandas as pd
from pylab import mpl, plt
import os
import json

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

import time
from sklearn.preprocessing import MinMaxScaler
import torch
from rnn import RNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)


def import_data(data_name, dir_path, configs):
    # Check the configurations
    shift_time = configs['shift_time']

    dataPath = dir_path + data_name + '.csv'
    df = pd.read_csv(dataPath)

    # Change the start date
    if shift_time == '1hr':
        df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
        df = df.loc[df.date >= '2015-01-01']
    elif shift_time == '5min':
        df.datetime = pd.to_datetime(df.datetime, format='%Y-%m-%d %H:%M:%S')
        df = df.loc[df.datetime >= '2015-01-01 00:00:00']

    df = df.dropna()

    return df


def preprocess_data(df, configs):
    # Check the configurations
    w_hours = configs['w_hours']

    # Scale the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df['vol'] = scaler.fit_transform(df['vol'].values.reshape(-1, 1))

    if w_hours:
        df['hour'] = scaler.fit_transform(df['hour'].values.reshape(-1, 1))
        df = df[['hour', 'vol']]
    else:
        df = df[['vol']]

    df = df.fillna(method='ffill')

    return df


def load_data(stock, configs):
    # Check the configurations
    look_back = 12
    w_hours = configs['w_hours']

    # Convert to numpy array
    data_raw = stock.values

    # create all possible sequences of length look_back
    data = []
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data)

    print(data.shape)

    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-1, :]
    x_test = data[train_set_size:, :-1]
    if w_hours:
        y_train = data[:train_set_size, -1, :][:, 1][:, np.newaxis]
        y_test = data[train_set_size:, -1, :][:, 1][:, np.newaxis]
    else:
        y_train = data[:train_set_size, -1, :]
        y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]


def run_RNN(x_train, y_train_rnn, x_test, y_test_rnn, output_path, data_name, configs):
    # Check the configurations
    w_hours = configs['w_hours']
    input_dim = configs['input_dim']
    output_dim = configs['output_dim']
    hidden_dim = configs['hidden_dim']
    num_layers = configs['num_layers']
    learning_rate = configs['learning_rate']
    dropout = configs['dropout']
    model = configs['model']

    if w_hours:
        input_dim += 1

    # Set the model
    model = RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                device=device, dropout=dropout, model=model).to(device)
    criterion = torch.nn.MSELoss(reduction='mean')

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(model)

    # Train model
    num_epochs = 1200

    hist = np.zeros(num_epochs)
    start_time = time.time()
    for t in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_rnn)
        if t % 100 == 0 and t != 0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    training_time = time.time() - start_time
    print("Training time: {}".format(training_time))

    fig, ax = plt.subplots()
    start = 100
    end = 200
    x = np.arange(y_train_pred.shape[0])[start:end]
    y_pred = y_train_pred.cpu().detach().numpy()[:, 0][start:end]
    y_real = y_train_rnn.cpu().detach().numpy()[:, 0][start:end]
    ax.plot(x, y_pred, label='pred')
    ax.plot(x, y_real, label='real')
    ax.legend()
    plt.savefig('plot_out/' + output_path + '/' + data_name + '.jpg')
    plt.show()

    # Test the result
    y_test_pred = model(x_test)
    loss = criterion(y_test_pred, y_test_rnn)
    MSE = round(loss.item(), 6)
    print("Test MSE: ", MSE)

    # Save the test results
    x = np.arange(y_test_pred.shape[0]).tolist()
    y_test_pred = y_test_pred.cpu().detach().numpy()[:, 0].tolist()
    y_test_rnn = y_test_rnn.cpu().detach().numpy()[:, 0].tolist()

    df = pd.DataFrame(
        {"real": y_test_rnn,
         "pred": y_test_pred
         })

    df.to_csv(
        './data_out/' + output_path + '/' + data_name + '_MSE_' + str(MSE) + '.csv')

    return MSE


def run_forecast(company, configs, dir_path):
    output_path = '5yrs_shiftTime_' + configs['shift_time'] + '_model_' + configs['model'] + '_wHours_' + \
                  str(configs['w_hours']) + '_numLayers_' + str(configs['num_layers']) + '_hiddenDim_' + \
                  str(configs['hidden_dim'])
    print('plot_out directory gen', os.system('mkdir plot_out\\' + output_path))
    print('data_out directory gen', os.system('mkdir data_out\\' + output_path))

    data_name = company + '_' + configs['shift_time'] + configs['additional_info']
    df = import_data(data_name, dir_path, configs)

    df = preprocess_data(df, configs)

    x_train, y_train, x_test, y_test = load_data(df, configs)
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)

    # Change to tensor
    x_train = torch.from_numpy(x_train).type(torch.Tensor).to(device)
    x_test = torch.from_numpy(x_test).type(torch.Tensor).to(device)
    y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)
    y_test = torch.from_numpy(y_test).type(torch.Tensor).to(device)

    data_name = data_name

    MSE = run_RNN(x_train, y_train, x_test, y_test, output_path, data_name, configs)

    return MSE

def draw_plot(company_name, hyper_parameter, plot_data):

    point_list = ['C0o-', 'C1o-', 'C2o-', 'C3o-', 'C4o-', 'C5o-']
    x = np.arange(len(plot_data[0][1]))
    x_label = np.zeros(len(plot_data[0][1]))
    for i, val in enumerate(plot_data[0][1]):
        x_label[i] = val[0]

    fig, ax = plt.subplots(1, 1)
    for j, model in enumerate(plot_data):
        model_name = model[0]
        y = np.zeros(len(model[1]))
        for i, val in enumerate(model[1]):
            y[i] = val[1]

        print(company_name, model_name, hyper_parameter)
        print(x)
        print(y)

        ax.plot(x, y, point_list[j], label=model_name)

    ax.set_title(company_name + "'s MSE results w/ " + hyper_parameter, fontdict={'fontsize': 18, 'weight': 'bold'})
    ax.set_ylabel('MSE', fontdict={'fontsize': 15, 'weight': 'bold'})
    ax.set_xlabel(hyper_parameter, fontdict={'fontsize': 15, 'weight': 'bold'})
    ax.set_xticks(x)
    ax.set_xticklabels(x_label)
    ax.legend()

    dir_path = 'MSE_figures\\'+company_name
    os.system('mkdir '+dir_path)
    plt.savefig(dir_path+'\\'+company_name+'_'+hyper_parameter+'.png')


if __name__ == '__main__':
    i = 0

    # Initialization of the data
    # model: 'LSTM', 'GRU'
    # shift_time: '1hr', '5min'
    # additional_info: '', '_GARCH', '_NEWS', '_GARCH_NEWS'
    company_list = ['AAPL', 'IBM', 'JNJ', 'VZ', 'XOM']
    dir_path = 'data/'
    configs = {'input_dim': 1, 'output_dim': 1, 'hidden_dim': 128, 'num_layers': 3,
               'learning_rate': 0.000015, 'dropout': 0.05, 'w_hours': True, 'model': 'GRU',
               'shift_time': '1hr', 'additional_info': ''}

    # Generate the test options
    model_types = ['LSTM', 'GRU']
    option_list = {}
    option_list['hidden_dim'] = [32, 64, 128, 256]
    option_list['num_layers'] = [2, 3, 4, 5]
    option_list['dropout'] = [0.01, 0.05, 0.1, 0.2]

    # Run and generates the volatility prediction MSE values for each option combination
    data_per_company = []
    for company in company_list:
        data_per_model = []
        for model in model_types:
            MSE_data = {}
            for key in option_list.keys():
                MSE_data[key] = []
                for option in option_list[key]:
                    test_configs = configs.copy()
                    test_configs[key] = option
                    test_configs['model'] = model

                    if test_configs['hidden_dim'] == 16:
                        test_configs['learning_rate'] = 0.00005
                    elif test_configs['hidden_dim'] == 32:
                        test_configs['learning_rate'] = 0.00002
                    elif test_configs['hidden_dim'] == 64:
                        test_configs['learning_rate'] = 0.000015
                    elif test_configs['hidden_dim'] == 128:
                        test_configs['learning_rate'] = 0.000005
                    elif test_configs['hidden_dim'] == 256:
                        test_configs['learning_rate'] = 0.000003


                    MSE = run_forecast(company, test_configs, dir_path)
                    #MSE = i
                    i += 1

                    MSE_data[key].append([option, MSE])

            data_per_model.append([model, MSE_data])

        data_per_company.append([company, data_per_model])

    a_file = open('data_bf_process.json', 'w')
    json.dump(data_per_company, a_file)
    a_file.close()

    # Generate the plot based on the MSE data
    for d1 in data_per_company:
        company_name = d1[0]
        plot_data_list = {}
        for d2 in d1[1]:
            model_name = d2[0]
            for key in d2[1].keys():
                hyper_parameter_name = key
                try:
                    plot_data_list[hyper_parameter_name].append([model_name, d2[1][key]])
                except:
                    plot_data_list[hyper_parameter_name] = [[model_name, d2[1][key]]]

        # Draw the graph
        print(plot_data_list)
        for hyper_parameter in plot_data_list.keys():
            draw_plot(company_name, hyper_parameter, plot_data_list[hyper_parameter])
