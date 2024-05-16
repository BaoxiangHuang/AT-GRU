import pandas as pd
from torch.utils.data import DataLoader
from preprocess import ChlDataset, DatasetProcess
from model import *
import argparse
import os
import torch.nn as nn
from torch.optim import RMSprop, Adam
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import orjson
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,mean_absolute_percentage_error

parser = argparse.ArgumentParser(description='Chl Prediction')
parser.add_argument('--epochs', default=1000, type=int, help='the train epochs of net work')
parser.add_argument('--lr', default=0.00001, type=float, help='the learning rate of the net work')
parser.add_argument('--rnn_number_layer', default=4, type=int, help='the number of the GRU/LSTM layer ')
parser.add_argument('--rnn_hidden_size', default=64, type=int, help='the hidden size of GRU or LSTM')
parser.add_argument('--encoder_input_size', default=16 * 4 + 4, type=int, help='the input size of encoder')
parser.add_argument('--encoder_hidden_size', default=64, type=int, help='the hidden size of encoder')
parser.add_argument('--decoder_output_size', default=3, type=int, help='the mlp decoder hidden size')
parser.add_argument('--output_size', default=2, type=int, help='the predict value number')
parser.add_argument('--batch_size', default=6, type=int, help='the batch size of train loader')

args = parser.parse_args()

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = AT_GRU(encoder_input_size=args.encoder_input_size, num_layers=args.rnn_number_layer,rnn_hidden_size=args.rnn_hidden_size,encoder_hidden_size=args.encoder_hidden_size ).to(device)
    loss_func = nn.L1Loss().to(device)
    optimizer = Adam(net.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    if not os.path.exists('data/data.json'):
        dataset = DatasetProcess()
        dataset.save()
    train_dataloader = DataLoader(ChlDataset(train=True), args.batch_size, shuffle=True)
    test_dataloader = DataLoader(ChlDataset(train=False), args.batch_size, shuffle=False)
    mean = ChlDataset(train=True).mean
    std = ChlDataset(train=True).std
    train_losses = []
    test_losses = []
    for epoch in range(args.epochs):
        epoch_loss = torch.zeros(1)
        for i, (data, label) in enumerate(train_dataloader):
            output = net(data.to(device))
            optimizer.zero_grad()
            loss = loss_func(output, label[:, :300].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(train_loss.item())
        scheduler.step(epoch_loss)
        print(f'Processing: [{epoch+1} / {args.epochs}] | Loss: {round((epoch_loss/len(train_dataloader)).item(), 6)} | Learning Rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        if (epoch + 1) % 10 == 0:
            prediction = []
            test_loss = torch.zeros(1)
            net.eval()
            with torch.no_grad():
                for i, (data, label) in enumerate(test_dataloader):
                    output = net(data.to(device))
                    loss = loss_func(output, label[:, :300].to(device))
                    test_loss += loss.item()
                    output = output.detach().cpu().numpy()
                    label = label.cpu().numpy()
                    output[:, :300] = output[:, :300] * std[:300] + mean[:300]
                    label[:, :300] = label[:, :300] * std[:300] + mean[:300]
                    for j in range(label.shape[0]):
                        prediction.append({'label': label[j, :300].tolist(), 'predict': output[j, ...].tolist()})
                test_loss= test_loss / len(test_dataloader)
                test_losses.append(test_loss.item())
                print(len(test_loss),len(test_loss))
                plt.plot(range(1, epoch + 2), train_losses, label='Train Loss')
                plt.plot(range(10,epoch + 2, 10), test_losses, label='Test Loss')
                png = 'loss_tmp_' + str(epoch+1) + '.png'
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Train and Test Loss')
                plt.legend()
                plt.savefig(png)
                plt.show()
                checkpoint_path = f'weights/epoch_{epoch + 1}.pth'
                torch.save(net.state_dict(), checkpoint_path)
                print(f'Saved checkpoint at {checkpoint_path}')
                file_name = f'result/{epoch}.json'
                with open(file_name, 'w') as f:
                    f.write(orjson.dumps(prediction).decode())
                f.close()
                print(f'Processing: [{epoch+1} / {args.epochs}] | Test Loss: {round((test_loss).item(), 6)}')
                y_true = np.array(label[:,:300])
                y_pred = np.array(output[:, :300])
                test_mse = []
                test_mse.append(mean_squared_error(y_true, y_pred))
                print('MSE:', mean_squared_error(y_true, y_pred))
                test_rmse = []
                test_rmse.append(np.sqrt(mean_squared_error(y_true, y_pred)))
                print('RMSE:', np.sqrt(mean_squared_error(y_true, y_pred)))
                test_mae = []
                test_mae.append(mean_absolute_error(y_true, y_pred))
                print('MAE:',mean_absolute_error(y_true, y_pred))
                test_r2 = []
                score = r2_score(y_true.reshape(-1), y_pred.reshape(-1))
                test_r2.append(score)
                print('R2：:', score)
                print('------------------------------------')
            net.train()
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1, 10), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()