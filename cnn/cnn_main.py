#!/apps/anaconda3/bin/python

import time
import torch
import os
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cnn import *
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)


def splitSequence(seq, n_ahead):
    x, y = [], []
    for i in range(len(seq)):
        end_idx = i + n_ahead
        
        if end_idx > len(seq)-1:
            break
        seq_x, seq_y = seq[i:end_idx], seq[end_idx]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

def trainModel(train_losses):
    running_loss = 0.0
    
    model.train()
    
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(inputs.float())
        loss = criterion(preds, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss
        
    train_loss = running_loss/len(train_loader)
    train_losses.append(train_loss.detach().numpy())

    return train_losses


if __name__ == '__main__':
    #, 200, 3
    ticker, train_epochs, n_ahead = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    df = pd.read_csv('/work/pl2669/taq_project/data/agg_1hr/' + ticker + '_1hr.csv')
    df = df.dropna(subset=['vol'])
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
    df_train = df.loc[(df.date >= '2015-01-01') & (df.date < '2019-01-01')].copy()
    df_test = df.loc[df.date >= '2019-01-01'].copy()

    #need to do or we get batches with different dimensions
    while df_train.shape[0] % n_ahead != 1:
        df_train = df_train[:-1].copy()

    x_train, y_train = splitSequence(df_train.vol.values, n_ahead) 

    train = VolDataset(x_train.reshape(x_train.shape[0], x_train.shape[1], 1), y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=False)
    
    model = CNN(3, 64, 128, 50, 50, 2, device)     
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()
   
    ## Training
    train_losses = []

    epochs = train_epochs
    
    #pbar = tqdm(np.arange(1, epochs, 1))
    #for epoch in pbar:
    for epoch in range(epochs):
        print('epochs {}/{}'.format(epoch+1,epochs))
        train_losses = trainModel(train_losses)
        print('MSE: ' + str(train_losses[-1].round(5)))
        #pbar.set_postfix_str('MSE: ' + str(train_losses[-1].round(5)))

    #plt.plot(train_losses, label='train_loss')
    #plt.title(ticker + ' MSE Loss')
    #plt.ylim(0, .2)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.savefig('/work/pl2669/taq_project_cnn/data/train_plots/' + ticker + '_mseloss.png')

    ## Testing
    x_test, y_test = splitSequence(df_test.vol.values, n_ahead)
    inputs = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    model.eval()
   
    batch_size = 2
    iters = int(inputs.shape[0] / batch_size)

    preds = []        
    for i in tqdm(range(iters)):
        model_input = torch.tensor(inputs[batch_size*i:batch_size*(i+1)]).float()
        pred = model(model_input)
        preds.append(pred.detach().numpy())

    preds = np.array(preds).reshape(1, -1)[0]
    
    #batch size can result in odd number
    y_test = y_test[:len(preds)]
    
    mse = np.mean((preds - y_test)**2)
    mae = np.mean(np.abs(preds - y_test))
     
    df_res = pd.DataFrame([preds, y_test]).T
    df_res.columns = ['pred', 'true']
    df_res.to_csv('/work/pl2669/taq_project_cnn/data/pred_mse/' + ticker + '_predres.csv',index=False)


    mse_file = open('/work/pl2669/taq_project_cnn/data/pred_mse/' + ticker + '_pred_mseNmae.txt', 'w')
    mse_file.write(ticker + ' MSE: ' + str(mse))
    mse_file.write(ticker + ' MAE: ' + str(mae))
    mse_file.close()
