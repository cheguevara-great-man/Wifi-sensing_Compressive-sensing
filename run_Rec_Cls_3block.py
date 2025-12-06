import numpy as np
import torch
import torch.nn as nn
import argparse
from util import load_data_n_model

import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
log_path = './checkpoints'
folder_name = "UT_HAR_Rec_Cls_3block"

def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tensor_loader:
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss/len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy/len(tensor_loader)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))

        torch.save(model.state_dict(), os.path.join(log_path, folder_name, '{}_Rec_Cls_3block_MODEL.pth'.format(epoch + 1)))

    return

def load_checkpoint(model, checkpoint_path, device):

    #net.load_state_dict(torch.load(pth_filepath, map_location='cuda:0')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print('模型加载好了！')
    #print(f"✅ 模型加载好了: {checkpoint_path} (epoch={checkpoint['epoch']})")
    return model

def test(model, tensor_loader, criterion, device, checkpoint_path='None', save_metrics_path='None'):
    #if checkpoint_path is not None:
     #   model = load_checkpoint(model, checkpoint_path, device)
    model.eval()
    test_acc = 0
    test_loss = 0
    all_labels = []
    all_preds = []
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predict_y.cpu().numpy())

        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
    return

    
def main():
    root = './Data/'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices = ['UT_HAR_data','NTU-Fi-HumanID','NTU-Fi_HAR','Widar'], default='UT_HAR_data')
    parser.add_argument('--model', choices = ['MLP','LeNet','ResNet18','ResNet50','ResNet101','RNN','GRU','LSTM','BiLSTM', 'CNN+GRU','ViT','LM','RecCls'], default='RecCls')
    args = parser.parse_args()

    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(
        model=model,
        tensor_loader= train_loader,
        num_epochs= train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device
         )


    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device= device,
        checkpoint_path='./checkpoints/UT_HAR_Rec_Cls_3block/67_Rec_Cls_MODEL.pth',
        save_metrics_path='./Val/UT_HAR_Rec_Cls_3block/test_metrics.txt'
        )
    return


if __name__ == "__main__":
    main()
