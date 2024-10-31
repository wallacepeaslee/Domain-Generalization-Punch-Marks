# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:55:26 2023

@author: Wallace Peaslee
"""



import torchvision.transforms as transforms
from torchvision import models
import torch
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from PIL import Image
import time
import copy
#import numpy as np
#from torchvision.io import read_image


labelDict = {}
for i in range(17):
    labelDict[i] = i


class CustomDataset(Dataset):
    def __init__(self, imagePaths, transform = None):
        self.images = []
        self.labels = []
        self.transform = transform
        imagePathsFile = open(imagePaths, 'r')
        imagePathsFileLines = imagePathsFile.readlines()
        for line in imagePathsFileLines:
            currLabel = int(line.split(' ')[1])
            self.images.append(line.split(' ')[0])
            self.labels.append(labelDict[currLabel])
        imagePathsFile.close()
            
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        label = self.labels[idx]
        imagePath = self.images[idx]
        #imagePIL = read_image(imagePath)
        imagePIL = Image.open(imagePath)
        #sample = {"image": imagePIL, "class": label}
        if self.transform:
            imagePIL = self.transform(imagePIL)
        return imagePIL, label


def runValidExperiment(trainTxtFilePath, validTxtFilePath, mcbTxtFilePath, FitzTxtFilePath, CologneTxtFilePath, savePath, trialNums):    
    numClasses = 17
    batchSize = 32
    
    # augment_transform = transforms.Compose([
    #     # transforms.Resize((224,224)),
    #     transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio = (0.95, 1.05)),
    #     transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
    #     transforms.RandomRotation(9),
    #     transforms.RandomGrayscale(0.4),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    
    # data_transform_paper = transforms.Compose([
    #     transforms.Resize((256,256)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.0,0.0,0.0], [1.0,1.0,1.0])
    #     ])
    
    data_transform_paper = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    trainSet = CustomDataset(trainTxtFilePath, transform = data_transform_paper)
    trainLoader = torch.utils.data.DataLoader(trainSet, shuffle=True, batch_size = batchSize)
    validSet = CustomDataset(validTxtFilePath, transform = data_transform_paper)
    validLoader = torch.utils.data.DataLoader(validSet, shuffle = False, batch_size = batchSize)
    
    mcbSet = CustomDataset(mcbTxtFilePath, transform = data_transform_paper)
    mcbLoader = torch.utils.data.DataLoader(mcbSet, shuffle = False, batch_size = batchSize)
    
    FitzSet = CustomDataset(FitzTxtFilePath, transform = data_transform_paper)
    FitzLoader = torch.utils.data.DataLoader(FitzSet, shuffle = False, batch_size = batchSize)
    
    CologneSet = CustomDataset(CologneTxtFilePath, transform = data_transform_paper)
    CologneLoader = torch.utils.data.DataLoader(CologneSet, shuffle = False, batch_size = batchSize)
    
    
    dataLoadersDict = {'train' : trainLoader,
                       'valid' : validLoader,
                       'mcb' : mcbLoader,
                       'fitz': FitzLoader,
                       'cologne': CologneLoader}
    dataSizesDict = {'train' : len(trainSet),
                       'valid' : len(validSet),
                       'mcb' : len(mcbSet),
                       'fitz': len(FitzSet),
                       'cologne': len(CologneSet)}
    print(dataSizesDict)
    
    model_ft = models.resnet50(pretrained=True)#(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    numFeaturesDefault = model_ft.fc.in_features
    
    model_ft.fc = nn.Linear(numFeaturesDefault, numClasses)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    
    lossFunction = nn.CrossEntropyLoss()
    optimizer_ft = optim.RAdam(model_ft.parameters())
    exp_lr_scheduler= lr_scheduler.MultiStepLR(optimizer_ft, [10, 15])
    
    model_ft, _, _ = train_model(model_ft, lossFunction, optimizer_ft, exp_lr_scheduler, dataLoadersDict, dataSizesDict, savePath, numEpochs = 20, trialInfo = trialNums)
    

def train_model(model, lossFunction, optimizer, scheduler, dataloaders, dataSizes, saveFolder, numEpochs = 20, trialInfo = [0,0]):
    trainSplit = trialInfo[0] 
    randomSplit = trialInfo[1]
    
    startTime = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('starting...', device, "cuda:0" if torch.cuda.is_available() else "cpu")
    
    currBestModel = copy.deepcopy(model.state_dict())
    currBestAccuracy = 0.0
    currBestEpoch = 0
    
    resStr = ''
    for epochNum in range(numEpochs):
        print("-"*10)
        print('Epoch:', epochNum)
        resStr += '\n---Epoch: ' + str(epochNum)
        for phase in dataloaders.keys():
            print(phase)
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            currRunLoss = 0.0
            currRunCorrect = 0
            
            for inputData, inputLabel in dataloaders[phase]:
                inputData = inputData.to(device)
                inputLabel = inputLabel.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    modelOutput = model(inputData)
                    _, predictions = torch.max(modelOutput, 1)
                    currLoss = lossFunction(modelOutput, inputLabel)
                if phase == 'train':
                    currLoss.backward()
                    optimizer.step()
                    
                currRunLoss += currLoss.item() * inputData.size(0)
                currRunCorrect += torch.sum(predictions == inputLabel.data)    
            epochLoss = currRunLoss/dataSizes[phase]
            epochAccuracy = currRunCorrect.double()/dataSizes[phase]
            if phase == 'train':
                scheduler.step()
            if phase == 'valid' and epochAccuracy > currBestAccuracy:
                currBestAccuracy = epochAccuracy
                currBestModel = copy.deepcopy(model.state_dict())
                currBestEpoch = epochNum
            #print("Epoch loss:", epochLoss)
            #print("Epoch accuracy", epochAccuracy)
            print(f'Loss: {epochLoss:.4f}')
            print(f'Accuracy: {epochAccuracy:.4f}')
            resStr += '\n\t' + phase + f'\tLoss: {epochLoss:.4f}' + f'\tAccuracy: {epochAccuracy:.4f}'
            print('-'*10)
    print('Complete,', time.time() - startTime)
    print('best epoch:', currBestEpoch, currBestAccuracy)
    
    # torch.save(model.state_dict(), saveFolder + 'model_final.pth')
    finalWeights = copy.deepcopy(model.state_dict())
    model.load_state_dict(currBestModel)
    # torch.save(model.state_dict(), saveFolder + 'model_best.pth')
    
    resStr += '\n-----\n'
    resStr += 'best epoch: ' + str(currBestEpoch)
    resStr += '\nbest accuracy: ' + str(currBestAccuracy)
    
    resPath = saveFolder + "out-splt_" + str(trainSplit) + '_repitition_' + str(randomSplit) + '_info.txt'
    with open(resPath, 'a') as f:
        f.write(resStr)
    return model, finalWeights, currBestModel


if __name__ == "__main__":
    
    '''
    numDataSplits is how many different train/valid/test splits there will be.
    numTrials is how many times training and evaluation will be repeated for each
    train/valid/test split.
    '''
    numDataSplits = 3
    numTrials = 3
    
    '''
    There should be numDataSpits trainFilePaths and validFilePaths, which are paths
    to text files giving the training and validation data images.
    These text files should contain image filepaths and labels in the format:
        [filepath] [label]
    These filepaths should have
    '''
    trainFilePaths = ['' for _ in range(numDataSplits)]
    validFilePaths = ['' for _ in range(numDataSplits)]
    
    
    '''
    The below are test domain filepaths
    '''
    mcbFilePath = ''
    FitzFilePath = ''
    CologneFilePath = ''
    
    
    '''
    The below is a folder path where output information should be saved.
    '''
    outputSavePath = ''

    
    for dataSplit in range(numDataSplits):
        for randomTrial in range(numTrials):
            runValidExperiment(trainFilePaths[dataSplit],
                               validFilePaths[dataSplit],
                               mcbFilePath,
                               FitzFilePath,
                               CologneFilePath,
                               outputSavePath,
                               (dataSplit, randomTrial))
    
    

    
    
