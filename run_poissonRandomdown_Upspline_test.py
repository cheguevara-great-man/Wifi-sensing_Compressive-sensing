import numpy as np
import torch
import torch.nn as nn
import argparse
from util import load_data_n_model

import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
log_path = './checkpoints'
folder_name = "UT_HAR_poissonRandomdown_Upspline_ResNet50"


def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tensor_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss / len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy / len(tensor_loader)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch + 1, float(epoch_accuracy), float(epoch_loss)))

        torch.save(model.state_dict(),
                   os.path.join(log_path, folder_name, '{}_poissonRandomdown_Upspline_ResNet50_MODEL.pth'.format(epoch + 1)))

    return


def load_checkpoint(model, checkpoint_path, device):
    # net.load_state_dict(torch.load(pth_filepath, map_location='cuda:0')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print('模型加载好了！')
    # print(f"✅ 模型加载好了: {checkpoint_path} (epoch={checkpoint['epoch']})")
    return model


def test(model, tensor_loader, criterion, device, checkpoint_path='None', save_metrics_path='None', save_images_path='None', tsne_perplexity=30):
    if checkpoint_path is not None:
        model = load_checkpoint(model, checkpoint_path, device)
    model.eval()
    test_acc = 0
    test_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []
    all_features = []
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)

        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)

        loss = criterion(outputs, labels)
        predict_y = torch.argmax(outputs, dim=1).to(device)

        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.detach().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predict_y.cpu().numpy())
        feature = outputs  # 否则用 logits 作为特征
        all_features.extend(feature.detach().numpy())

        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc / len(tensor_loader)
    test_loss = test_loss / len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc), float(test_loss)))

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_features = np.array(all_features)
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    Sens = np.diag(cm) / np.sum(cm, axis=1)  # TP / (TP+FN)
    Spec = []
    for i in range(cm.shape[0]):
        tn = np.sum(np.delete(np.delete(cm, i, 0), i, 1))
        fp = np.sum(np.delete(cm, i, 0)[:, i])
        Spec.append(tn / (tn + fp))
    Spec = np.array(Spec)
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Sensitivity': Sens,
        'Specificity': Spec
    }
    print(f"✅ Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}")
    print(f"Sensitivity per class: {Sens}")
    print(f"Specificity per class: {Spec}")
    # 画混淆矩阵
    plt.figure(figsize=(8, 6))
    class_names = ['0','1','2','3','4','5','6']
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_images_path, 'UT_HAR_poissonRandomdown_Upspline_ResNet50_confusion.png'))
    plt.show()

    print('********************混淆矩阵画图完毕******************************')

    # 画ROC和AUC曲线
    class_names = ['0','1','2','3','4','5','6']
    n_classes = 7
    all_labels_bin = label_binarize(all_labels, classes=np.arange(n_classes))  # one-hot
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(save_images_path, 'UT_HAR_poissonRandomdown_Upspline_ResNet50_AUC_ROC.png'))
    plt.show()
    print('********************ROC和AUC画图完毕******************************')

    #画混淆矩阵
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
    tsne_results = tsne.fit_transform(all_features)

    plt.figure(figsize=(8, 6))
    for idx, label in enumerate(np.unique(all_labels)):
        plt.scatter(tsne_results[all_labels == label, 0],
                    tsne_results[all_labels == label, 1],
                    label=class_names[idx], alpha=0.7)
    plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    #plt.grid(True)
    plt.savefig(os.path.join(save_images_path, 'UT_HAR_poissonRandomdown_Upspline_ResNet50_Tsne.png'))
    plt.show()
    print('********************Tsne画图完毕******************************')

    print('********************以下是保存指标******************************')
    if save_metrics_path is not None:
        os.makedirs(os.path.dirname(save_metrics_path), exist_ok=True)
        with open(save_metrics_path, 'w') as f:
            f.write("Classification Metrics\n")
            f.write("====================\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1: {f1:.4f}\n")
            f.write("\nSensitivity per class:\n")
            for idx, s in enumerate(Sens):
                f.write(f"  Class {idx}: {s:.4f}\n")
            f.write("\nSpecificity per class:\n")
            for idx, s in enumerate(Spec):
                f.write(f"  Class {idx}: {s:.4f}\n")
        print(f" 分类指标已保存为txt: {save_metrics_path}")

    return


def main():
    root = './Data/'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices=['UT_HAR_data', 'NTU-Fi-HumanID', 'NTU-Fi_HAR', 'Widar'],
                        default='UT_HAR_data')
    parser.add_argument('--model',
                        choices=['MLP', 'LeNet', 'ResNet18', 'ResNet50', 'ResNet101', 'RNN', 'GRU', 'LSTM', 'BiLSTM',
                                 'CNN+GRU', 'ViT'], default='ResNet50')
    args = parser.parse_args()

    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device,
        checkpoint_path='./checkpoints/UT_HAR_poissonRandomdown_Upspline_ResNet50/200_poissonRandomdown_Upspline_ResNet50_MODEL.pth',
        save_metrics_path='./Val/UT_HAR_poissonRandomdown_Upspline_ResNet50/test_metrics.txt',
        save_images_path = './Val/UT_HAR_poissonRandomdown_Upspline_ResNet50/'
    )
    return


if __name__ == "__main__":
    main()
