import cv2
import math
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

import torch
import torchvision
import torchvision.models as models
import torchxrayvision as xrv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary

# Image preprocessing functions

def min_max_scaling(img, pretrained):
    """
    Scales image values between 0 and 1 and applies further scaling based on the pretraining model.
    
    Args:
        img (numpy array): Input image to be scaled.
        pretrained (str): Pretrained model type ('TorchX', 'ImageNet').
    
    Returns:
        img_normed (numpy array): Scaled image.
    """
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    
    if pretrained == 'TorchX':
        # Scale the image pixel values to roughly range [-1024, 1024]
        img_normed = (2 * img - 1.0) * 1024 
    elif pretrained == 'ImageN':
        # Scale the image pixel values to roughly range [-0.5, 0.5]
        img_normed = img - 0.5

    return img_normed

def rotate_images(image, img_size):
    """
    Rotates image by a random angle between -15 and 15 degrees.
    
    Args:
        image (numpy array): Image to be rotated.
        img_size (int): Size of the output image.
    
    Returns:
        rotated (numpy array): Rotated image.
    """
    scale = 1.0
    center = (img_size / 2, img_size / 2)
    angle = random.randint(-15, 15)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (img_size, img_size), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return rotated

def translation(image):
    """
    Translates image by a random amount in both x and y directions.
    
    Args:
        image (numpy array): Input image to be translated.
    
    Returns:
        translate (numpy array): Translated image.
    """
    x = random.randint(-15, 15)
    y = random.randint(-15, 15)
    rows, cols, z = image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    translated = cv2.warpAffine(image, M, (cols, rows))
    
    return translated

def apply_aug(image, img_size):
    """
    Applies augmentation to the image: scaling, rotation, and translation.
    
    Args:
        image (numpy array): Input image to be augmented.
        img_size (int): Size of the image.
    
    Returns:
        image (numpy array): Augmented image.
    """
    image[:, :, 0] = image[:, :, 0].astype(np.float32)
    image[:, :, 0] = (image[:, :, 0] - image[:, :, 0].min()) / (image[:, :, 0].max() - image[:, :, 0].min())
    image = rotate_images(image, img_size)
    return image

def oversample_images(X, y, img_size): 
    """
    Oversamples the minority class in the dataset using image augmentations.
    
    Args:
        X (numpy array): Input image data.
        y (numpy array): Corresponding labels.
        img_size (int): Target size of images after augmentation.
    
    Returns:
        X, y (numpy arrays): Augmented dataset.
    """
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    new_img = []
    new_label = []
    
    if n_neg > n_pos:
        minority_label, majority_label = 1, 0
        required_augmentations = n_neg - n_pos
    else:
        minority_label, majority_label = 0, 1
        required_augmentations = n_pos - n_neg

    counter = 0
    while counter < required_augmentations:
        for i in range(len(y)):
            if y[i] == minority_label:
                # Number of augmentations per minority image
                n_augmentations = math.floor(required_augmentations / min(n_pos, n_neg))
                for j in range(n_augmentations):
                    if counter >= required_augmentations:
                        break
                    image = apply_aug(X[i], img_size)
                    new_img.append(image)
                    new_label.append(y[i])
                    counter += 1

    new_img = np.array(new_img)
    new_label = np.array(new_label)
    X = np.append(X, new_img, axis=0)
    y = np.append(y, new_label)

    return X, y

def clahe(X_bbox, cl, gs):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast.
    
    Args:
        X_bbox (numpy array): Images to be enhanced.
        cl (float): Clip limit for CLAHE.
        gs (int): Grid size for CLAHE.
    
    Returns:
        X_histeq (list): List of enhanced images.
    """
    clahe = cv2.createCLAHE(clipLimit=cl * gs * gs, tileGridSize=(gs, gs))
    X_histeq = []
    for img in X_bbox:
        img = img.astype('uint8')
        img = clahe.apply(img)
        X_histeq.append(img)
    return X_histeq

def sharpening(X_bbox):
    """
    Applies image sharpening using Gaussian blur and unsharp masking.
    
    Args:
        X_bbox (numpy array): Images to be sharpened.
    
    Returns:
        X_sharp (list): List of sharpened images.
    """
    X_sharp = []
    for img in X_bbox:
        img = img.astype('uint8')
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        unsharp_image = cv2.addWeighted(img, 3.0, blur, -2.0, 0)
        X_sharp.append(unsharp_image)
    return X_sharp

def bbox2_square(img):
    """
    Finds the smallest square bounding box around the non-zero region of the input image.
    Returns the ymin, ymax, xmin, and xmax coordinates.
    """
    # Get rows and columns where there are non-zero pixels
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    # Find the bounding box coordinates
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    # Compute the height and width of the bounding box
    ylen = ymax - ymin
    xlen = xmax - xmin

    diff = ylen - xlen
    if diff > 0:
        xmin = xmin - round(diff/2)
        xmax = xmax + round(diff/2)
        if xmin < 0:
            xmin = 0
            xmax = ylen - 1
        if xmax > 511:
            xmax = 511
            xmin = 511 - ylen + 1
    elif diff < 0: 
        ymin = ymin - round(-diff/2)
        ymax = ymax + round(-diff/2)
        if ymin < 0:
            ymin = 0
            ymax = xlen - 1
        if ymax > 511:
            ymax = 511
            ymin = 511 - xlen + 1

    return int(ymin), int(ymax), int(xmin), int(xmax)
    
def preprocess_seg(X, y, sharpen, histeq, cl, gs, pretrained):
    """
    Preprocesses a set of images by extracting a square region of interest (bounding box),
    resizing it, applying optional histogram equalization and sharpening, and scaling for
    further use in a machine learning model.

    Args:
        X (ndarray): Input images of shape (n_samples, height, width, channels).
        y (ndarray): Labels corresponding to the input images.
        sharpen (str): If 'True', applies sharpening to the images.
        histeq (str): If 'True', applies CLAHE histogram equalization to the images.
        cl (int): Clip limit for CLAHE.
        gs (tuple): Grid size for CLAHE.
        pretrained (str): Specifies the pretraining method, used for scaling (e.g., 'TorchX' or 'ImageNet').

    Returns:
        tuple: Processed images and labels, where the images are resized, preprocessed, 
               and scaled (shape: [n_samples, img_size, img_size, 1]).
    """
    X_bbox = []
    img_size = 512
    for i in range(len(X)):
        ymin, ymax, xmin, xmax = bbox2_square(X[i,:,:,1])
        img = X[i,:,:,0].astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        bbox = img[ymin:ymax+1, xmin:xmax+1]
        bbox = cv2.resize(bbox, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
        X_bbox.append(bbox)
     
    X_bbox, y = shuffle(X_bbox, y)
    X_bbox = np.array(X_bbox)
    # print('after bbox',np.min(X_bbox),np.max(X_bbox))
    
    if histeq == 'True':
        X_histeq = clahe(X_bbox, cl, gs)
        X_bbox = np.array(X_histeq)
        # print('after clahe',np.min(X_bbox),np.max(X_bbox))
        
    else: 
        X_bbox = np.array(X_bbox)   

    if sharpen == 'True':
        X_sharp = sharpening(X_bbox)
        X_bbox = np.array(X_sharp)
    else:
        X_bbox = np.array(X_bbox)

    X_processed = []
    for i in range(len(X_bbox)):
        bbox = min_max_scaling(X_bbox[i], pretrained)
        X_processed.append(bbox)

    print('after processing',np.min(X_processed),np.max(X_processed))

    X_processed = np.expand_dims(X_processed, axis=-1)
    
    return X_processed, y

# Model 

def load_model(pretrained):
    """
    Loads and modifies a pretrained model for binary classification.
    """
    if pretrained == 'TorchX':
        # Load TorchXRayVision pretrained model
        model = xrv.models.ResNet(weights="resnet50-res512-all")
        base_model = model.model
        # Modify the final fully connected layer for binary classification
        num_ftrs = base_model.fc.in_features
        base_model.fc = torch.nn.Linear(num_ftrs, 2048)
    
    elif pretrained == 'ImageN':
        # Load ImageNet Pre-trained Model in PyTorch
        base_model = models.resnet50(num_classes=1000, weights=True)
        # Modify the first convolutional layer to accept 1 channel input
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # freeze all layers but the last fc
    for name, param in base_model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # unfreeze the last convolutional layers
    for name, param in base_model.named_parameters():
        if  "layer4.2.conv3" in name or "layer4.2.bn3" in name:
            param.requires_grad = True

    class ExtendedModel(nn.Module):
        def __init__(self, base_model):
            super(ExtendedModel, self).__init__()
            self.base_model = base_model
            self.dropout = nn.Dropout(0.3)
            self.dense1 = nn.Linear(2048, 32)
            self.batchnorm = nn.BatchNorm1d(32)
            self.relu = nn.ReLU()
            self.dense2 = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            x = self.base_model(x)
            x = self.dropout(x)
            x = self.dense1(x)
            x = self.batchnorm(x)
            x = self.relu(x)
            x = self.dense2(x)
            return x

    class ExtendedModel_imagenet(nn.Module):
        def __init__(self, base_model):
            super(ExtendedModel_imagenet, self).__init__()
            self.base_model = base_model
            self.dropout = nn.Dropout(0.3)
            self.dense1 = nn.Linear(1000, 32)
            self.batchnorm = nn.BatchNorm1d(32)
            self.relu = nn.ReLU()
            self.dense2 = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            x = self.base_model(x)
            x = self.dropout(x)
            x = self.dense1(x)
            x = self.batchnorm(x)
            x = self.relu(x)
            x = self.dense2(x)
            x = self.sigmoid(x)
            return x

    # Create an instance of the extended model
    if pretrained == 'ImageN':
        extended_model = ExtendedModel_imagenet(base_model)
    else:
        extended_model = ExtendedModel(base_model)

    return extended_model

def create_data_loader(x_data, y_data, batch_size):
    """
    Creates a DataLoader for the provided input and target data.
    """
    tensor_x = torch.Tensor(x_data) 
    tensor_y = torch.Tensor(y_data)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model saving and performance visualization functions

class History_trained_model(object):
    """
    Stores the training history, epoch, and parameters of a model.
    """
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params

def save_hist(history, fold_no, histpath):
    """
    Saves model history and plots accuracy/loss graphs for the given fold.
    """
    # Save the history object
    with open(histpath + '/fold_' + str(fold_no) + '_history.pickle', 'wb') as file:
        model_history = History_trained_model(history, None, None)  # Adjust as needed
        pickle.dump(model_history, file, pickle.HIGHEST_PROTOCOL)

    # Plot and save accuracy graph
    plt.plot(history['train_accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(f"Fold#{fold_no} Model Accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(histpath + f"/fold_{fold_no}_acc.png", bbox_inches="tight")
    plt.clf()

    # Plot and save loss graph
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title(f"Fold#{fold_no} Model Loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(histpath + f"/fold_{fold_no}_loss.png", bbox_inches="tight")
    plt.clf()

def save_roc(y_test, y_pred, fold_no, histpath):
    """
    Plots and saves ROC curve for the given fold, including the AUC score.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.axis([0, 1, 0, 1])
    plt.title(f"Fold#{fold_no} ROC Curve (AUC= {auc_score:.5f})")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(histpath + f"/fold_{fold_no}_roc.png", bbox_inches="tight")
    plt.clf()

def save_aveconf(confusion_list, histpath):
    """
    Saves the average normalized confusion matrix across k-folds.
    """
    conf_mat_mean = np.mean(confusion_list, axis=0)
    conf_mat_norm = np.around(conf_mat_mean / conf_mat_mean.sum(axis=1)[:, np.newaxis], decimals=2)

    fig, ax = plt.subplots()
    cmap = 'Blues'
    cax = ax.matshow(conf_mat_norm, cmap=cmap)
    plt.colorbar(cax)

    # Set labels
    ax.xaxis.set_major_locator(ticker.FixedLocator([0, 1]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([0, 1]))  
    ax.set_xticklabels(['non-ICU', 'ICU'], fontsize=12)
    ax.set_yticklabels(['non-ICU', 'ICU'], fontsize=12)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)

    # Add annotations
    for (i, j), val in np.ndenumerate(conf_mat_norm):
        text_color = 'white' if conf_mat_norm[i, j] > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=12)

    ax.xaxis.tick_bottom()  # Move x-axis ticks to the bottom
    plt.savefig(histpath + "/confusion.png", bbox_inches="tight")
    plt.clf()


# For testing
def load_model_weight(weightPath, pretrained):
    if pretrained == 'TorchX': 
        model = models.resnet50(num_classes=2048, weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  
        class ExtendedModel(nn.Module):
            def __init__(self, base_model):
                super(ExtendedModel, self).__init__()
                self.base_model = base_model
                self.dropout = nn.Dropout(0.3)
                self.dense1 = nn.Linear(2048, 32)
                self.batchnorm = nn.BatchNorm1d(32)
                self.relu = nn.ReLU()
                self.dense2 = nn.Linear(32, 1)
                # self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                x = self.base_model(x)
                x = self.dropout(x)
                x = self.dense1(x)
                x = self.batchnorm(x)
                x = self.relu(x)
                x = self.dense2(x)
                # x = self.sigmoid(x)
                return x
            
        # Create an instance of the extended model
        model = ExtendedModel(model)

   
    elif pretrained == 'ImageN':
        # Load ImageNet Pre-trained Model in PyTorch
        base_model = models.resnet50(num_classes=1000, weights=False)
        # Modify the first convolutional layer to accept 1 channel input
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        class ExtendedModel_imagenet(nn.Module):
            def __init__(self, base_model):
                super(ExtendedModel_imagenet, self).__init__()
                self.base_model = base_model
                self.dropout = nn.Dropout(0.3)
                self.dense1 = nn.Linear(1000, 32)
                self.batchnorm = nn.BatchNorm1d(32)
                self.relu = nn.ReLU()
                self.dense2 = nn.Linear(32, 1)
                # self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                x = self.base_model(x)
                x = self.dropout(x)
                x = self.dense1(x)
                x = self.batchnorm(x)
                x = self.relu(x)
                x = self.dense2(x)
                # x = self.sigmoid(x)
                return x
        model = ExtendedModel_imagenet(base_model)

    state_dict = torch.load(weightPath, map_location=torch.device('cuda'))
    model.load_state_dict(state_dict)
    return model