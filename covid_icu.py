#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, accuracy_score, confusion_matrix)
from utils import *

start_time = time.time()

# Inputs - training parameters
sharpen, histeq_, pretrained, opt, lr, bs, ep = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
historyDir = sys.argv[8]
jobname = sys.argv[9]

# Expected histeq_ "False" or "True_{cl}_{gs}" clipLimit=cl*gs*gs, tileGridSize=(gs, gs)
if len(histeq_.split('_'))>1:
    histeq, cl, gs = histeq_.split('_')[0],  float(histeq_.split('_')[1]), int(histeq_.split('_')[2])
else:
    histeq, cl, gs = histeq_, 0, 0

# Prepare output folder and result file
histPath = os.path.join(historyDir, jobname)
if not os.path.exists(histPath):
    os.makedirs(histPath)

checkpoint_dir = os.path.join(histPath,"checkpoint")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Load data (nysbu-primary, ricord-secondary)
# data dimention: (N, 512, 512, 2)
nysbu_data = np.load('/path/to/nysbu_data_preprocessed.npy')
nysbu_label = pd.read_csv('/path/to/nysbu_label.csv')
ricord_data = np.load('/path/to/ricord_data_preprocessed.npy')
ricord_label = pd.read_csv('/path/to/ricord_label.csv')

nysbu_x = np.array(nysbu_data)
nysbu_y = np.array(nysbu_label['Label'].astype(int))
nysbu_id = np.array(nysbu_label['ID'])
ricord_y = np.array(ricord_label['Label'])

# Training testing split 
x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(nysbu_x, nysbu_y, nysbu_id, test_size = 0.2, stratify=nysbu_y, random_state=42)
x_test, y_test, id_test = preprocess_seg(x_test, y_test, sharpen, histeq, cl, gs, pretrained)
x_test = np.transpose(x_test, (0, 3, 1, 2))
print('Testing data distribution')
print('x_test', x_test.shape)
print('y_test', y_test.shape)
print("True: ", y_test.sum())
print("False: ",len(y_test) - y_test.sum())
# Expand training data with RICORD data
x_train = np.concatenate([ricord_data, x_train],0)
y_train = np.append(ricord_y, y_train, 0)
print('Training data distribution before data augmentation')
print('x_train', x_train.shape)
print('y_train', y_train.shape)
print("True: ", y_train.sum())
print("False: ",len(y_train) - y_train.sum())

ori_img_size = x_train.shape[1]

img_size = x_test.shape[3]
num_folds = int(kfold)
fold_no = 1
acc_list = []
loss_per_fold = []
confusion_list = []
precision_list = []
recall_list = []
specificity_list = []
macrof1_list = []
weightedf1_list = []
auc_list = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on :", device)


skfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
start_time = time.time()
# Loop for each fold
for train_idx, val_idx in skfold.split(x_train, y_train):
    
    # Data preparation
    x_training, y_training = oversample_images(x_train[train_idx], y_train[train_idx], ori_img_size)
    x_training, y_training = preprocess_seg(x_training, y_training, sharpen, histeq, cl, gs, pretrained)
    x_training = np.transpose(x_training, (0, 3, 1, 2))
    print('Training data distribution after data augmentation')
    print('x_training', x_training.shape)
    print('y_training', y_training.shape)
    print("True: ", y_training.sum())
    print("False: ",len(y_training) - y_training.sum())
    train_loader = create_data_loader(x_training, y_training, batch_size=int(bs))
    

    x_val, y_val = oversample_images(x_train[val_idx], y_train[val_idx], ori_img_size)
    x_val, y_val = preprocess_seg(x_val, y_val, sharpen, histeq, cl, gs, pretrained)
    x_val = np.transpose(x_val, (0, 3, 1, 2))
    print('Validation data distribution after data augmentation')
    print('x_val', x_val.shape)
    print('y_val', y_val.shape)
    print("True: ", y_val.sum())
    print("False: ",len(y_val) - y_val.sum())
    val_loader = create_data_loader(x_val, y_val, batch_size=int(bs))

    # Model
    extended_model = load_model(pretrained)
    model = extended_model.to(device)

    # Optimizer
    lr = float(lr)
    if opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    elif opt == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

    # ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-15)

    # Early Stopping Parameters
    early_stopping_patience = 50
    early_stopping_counter = 0
    best_val_loss = float('inf')

    # Loss function
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()

    print('------------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    # Training
    for epoch in range(int(ep)):
        model.train()
        train_loss = 0.0
        train_corrects = 0
        # Training loop (batch training)
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch).squeeze()
            output = torch.sigmoid(output)
            if torch.isnan(output).any():
                raise ValueError("NaN detected in output")
            loss = criterion(output, y_batch)
            train_loss += loss.item()
            train_corrects += (output.round() == y_batch).sum().item()
             # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_corrects / len(y_training)

        model.eval()
        # Validation loop
        val_loss = 0.0
        val_corrects = 0
        for x_val_batch, y_val_batch in val_loader:
            x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
            with torch.no_grad():
                val_outputs = model(x_val_batch).squeeze()
                val_outputs = torch.sigmoid(val_outputs)
                v_loss = criterion(val_outputs, y_val_batch)
                val_loss += v_loss.item()
                val_corrects += (val_outputs.round() == y_val_batch).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_corrects / len(y_val)
        print(f"Training Loss: {avg_train_loss}, Accuracy: {train_accuracy} || Validation Loss: {avg_val_loss}, Accuracy: {val_accuracy}")

        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        # Update learning rate
        scheduler.step(avg_val_loss) 
        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), f"{checkpoint_dir}/model_fold_{fold_no}_best.pt")
        elif epoch>=100:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                # Load the best model before testing
                model.load_state_dict(torch.load(f"{checkpoint_dir}/model_fold_{fold_no}_best.pt"))
                print("Early stopping triggered")
                break

    # Evaluate model on test data
    test_loader = create_data_loader(x_test, y_test, batch_size=int(bs))
    test_loss = 0.0
    test_corrects = 0
    y_pred_list = []  # List to store all predictions
    y_true_list = []  # List to store all true labels
    model.eval()
    for x_test_batch, y_test_batch in test_loader:
        x_test_batch, y_test_batch = x_test_batch.to(device), y_test_batch.to(device)
        with torch.no_grad():
            test_outputs = model(x_test_batch).squeeze()
            # fix the mismatch dimension
            if test_outputs.ndim != y_test_batch.ndim:
                print("test_outputs shape:", test_outputs.shape)
                print("y_test_batch shape:", y_test_batch.shape)
                print("Dimension mismatch detected. Adjusting dimensions.")
                test_outputs = test_outputs.unsqueeze(-1) if test_outputs.ndim < y_test_batch.ndim else test_outputs
                y_test_batch = y_test_batch.unsqueeze(-1) if y_test_batch.ndim < test_outputs.ndim else y_test_batch
            test_outputs = torch.sigmoid(test_outputs)
            t_loss = criterion(test_outputs, y_test_batch)
            test_loss += t_loss.item()
            test_corrects += (test_outputs.round() == y_test_batch).sum().item()
            y_pred_list.extend(test_outputs.cpu().detach().numpy())  # Convert predictions to probabilities
            y_true_list.extend(y_test_batch.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = test_corrects / len(y_test)
    print('------------------------------------------------------------------------------')
    print(f"Test Loss: {avg_test_loss}, Accuracy: {test_accuracy}")

    # Convert lists to numpy arrays for metric calculation
    y_pred_np = np.array(y_pred_list)
    y_true_np = np.array(y_true_list)

    # Calculate metrics
    confusion = confusion_matrix(y_true_np, y_pred_np >= 0.5)
    precision = precision_score(y_true_np, y_pred_np >= 0.5)
    recall = recall_score(y_true_np, y_pred_np >= 0.5)
    specificity = recall_score(~y_true_np.astype(bool), ~(y_pred_np >= 0.5))
    macro_f1 = f1_score(y_true_np, y_pred_np >= 0.5, average='macro')
    weighted_f1 = f1_score(y_true_np, y_pred_np >= 0.5, average='weighted')
    accuracy = accuracy_score(y_true_np, y_pred_np >= 0.5)
    auc = roc_auc_score(y_true_np, y_pred_np)
    Aacc = (recall+specificity)/2*100
    print(f"Recall: {recall}, Specificity: {specificity}")
    print(f"Aacc: {Aacc}, AUC: {auc}")
    # Append metrics to lists
    confusion_list.append(confusion)
    precision_list.append(precision)
    recall_list.append(recall)
    specificity_list.append(specificity)
    macrof1_list.append(macro_f1)
    weightedf1_list.append(weighted_f1)
    acc_list.append(accuracy * 100) 
    auc_list.append(auc)

    save_hist(history, fold_no, str(histPath))
    save_roc(y_true_np, y_pred_np, fold_no, str(histPath))
    save_aveconf(confusion_list, str(histPath))   
    
    # Save model checkpoint
    torch.save(model.state_dict(), f"{checkpoint_dir}/model_fold_{fold_no}.pt")

    fold_no += 1

# Calculate average metrics
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_specificity = np.mean(specificity_list)
avg_macro_f1 = np.mean(macrof1_list)
avg_weighted_f1 = np.mean(weightedf1_list)
avg_accuracy = np.mean(acc_list)
avg_Aacc = (avg_recall+avg_specificity)/2*100
avg_auc = np.mean(auc_list)

# Print results
print('------------------------------------------------------------------------------')
print('Average Precision:', avg_precision)
print('Average Recall:', avg_recall)
print('Average Specificity:', avg_specificity)
print('Average Macro F1 Score:', avg_macro_f1)
print('Average Weighted F1 Score:', avg_weighted_f1)
print('Average Accuracy:', avg_accuracy)
print('Average Class accuracy:', avg_Aacc)
print('Average AUC:', avg_auc)

elapsed_seconds = int(time.time() - start_time)
hours, remainder = divmod(elapsed_seconds, 3600)
minutes, seconds = divmod(remainder, 60)
formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
print(f"Elapsed time: {formatted_time}")
