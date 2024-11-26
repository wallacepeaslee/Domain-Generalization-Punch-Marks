# -*- coding: utf-8 -*-

import numpy as np
import os

def Stats(d):
    data = np.array(d)
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    return mean, err



def getDataDict(fp):
    res = {}
    currentEpoch = None
    bestEpoch = None
    bestAcc = None
    with open(fp, 'r') as f:
        for line in f:
            if 'Epoch:' in line:
                currentEpoch = int(line.strip().split(' ')[-1])
                res[currentEpoch] = {}
            if currentEpoch is not None:
                if 'valid' in line:
                    res[currentEpoch]['valid'] = float(line.strip().split(' ')[-1])
                if 'mcb' in line:
                    res[currentEpoch]['mcb'] = float(line.strip().split(' ')[-1])
                if 'fitz' in line:
                    res[currentEpoch]['fitz'] = float(line.strip().split(' ')[-1])
                if 'cologne' in line:
                    res[currentEpoch]['cologne'] = float(line.strip().split(' ')[-1])
                if '385' in line:
                    res[currentEpoch]['385'] = float(line.strip().split(' ')[-1])
            if 'best epoch' in line:
                bestEpoch = int(line.strip().split(' ')[-1])
            if 'best accuracy' in line:
                # print(line.strip().split(',')[-1])
                bestAcc = float(line.strip().split(',')[0].split('(')[1])
    return res, bestEpoch, bestAcc


def getDictMeans(dList):
    res = {}
    for elm in dList[0].keys():
        vals = []
        for d in dList:
            vals.append(d[elm])
        res[elm] = sum(vals)/len(vals)
    return res
            

def getStats(algDir):
    exps = os.listdir(algDir)
    trials = [[0 for _ in range(3)] for _ in range(3)]
    for exp in exps:
        expPath = os.path.join(algDir, exp)
        d, bestE, bestA = getDataDict(expPath)
        oracleAcc = d[19]['mcb']
        splitNum = int(exp.strip().split('_')[1])
        repNum = int(exp.strip().split('_')[3])
        trials[splitNum][repNum] = (d, (bestA, bestE), (oracleAcc, 19))
    
    
    #sorry for super unreadable code, I'm tired today
    oracleBest = [0,0,0]
    validBest = [0,0,0]
    for i in range(3):
        for j in range(3):
            if trials[i][j][2][0] > oracleBest[i]:
                oracleBest[i] = trials[i][j][2][0]
            if trials[i][j][1][0] > validBest[i]:
                validBest[i] = trials[i][j][1][0]
    
    
    resDataValid = []
    resDataOracle = []
    for i in range(3):
        currTrialsValid = []
        currTrialsOracle = []
        for j in range(3):
            if trials[i][j][1][0] == validBest[i]:
                currTrialsValid.append((trials[i][j][0], trials[i][j][1][1]))
            if trials[i][j][2][0] == oracleBest[i]:
                currTrialsOracle.append((trials[i][j][0], trials[i][j][2][1]))
        validList = [x[0][x[1]] for x in currTrialsValid]
        oracleList = [x[0][x[1]] for x in currTrialsOracle]
        
        #save just first...?
        validList = [validList[0]]
        oracleList = [oracleList[0]]
        
        validVals = getDictMeans(validList)
        oracleVals = getDictMeans(oracleList)
        
        resDataValid.append(validVals)
        resDataOracle.append(oracleVals)
    
    
    saveKeys = ['mcb', 'fitz', 'cologne', '385']
    resValid = {}
    resOracle = {}
    
    print('Valid:')
    for elm in resDataValid[0].keys():
        statVals = [x[elm] for x in resDataValid]
        print(elm, Stats(statVals))
        if elm in saveKeys:
            resValid[elm] = Stats(statVals)
    
    print('\nOracle:')
    for elm in resDataOracle[0].keys():
        statVals = [x[elm] for x in resDataOracle]
        print(elm, Stats(statVals))
        if elm in saveKeys:
            resOracle[elm] = Stats(statVals)
    
    return resValid, resOracle


def printAllStats():
    # folders = ['densenet121',
    #  'densenet121-aug',
    #  'densenet161',
    #  'densenet161-aug',
    #  'resnet18',
    #  'resnet18-aug',
    #  'resnet50',
    #  'resnet50-aug',]
    
    folders = ['resnet18-no_aug',
     'resnet18-aug',
     'resnet18-no_aug-no_pretraining',
     'resnet18-aug-no_pretraining',
     'resnet50-no_aug',
     'resnet50-aug',
     'resnet50-no_aug-no_pretraining',
     'resnet50-aug-no_pretraining',
     'densenet121-no_aug',
      'densenet121-aug',
      'densenet161-no_aug',
      'densenet161-aug']
    
    methods = ['resnet18',
     'resnet18*',
     'resnet18]',
     'resnet18*]',
     'resnet50',
     'resnet50*',
     'resnet50]',
     'resnet50*]',
     'densenet121',
      'densenet121*',
      'densenet161',
      'densenet161*']
    #TODO: update above with folder names and corresponding method names
    
    allValid = []
    allOracle = []
    for f in folders:
        print(f)
        tmpValid, tmpOracle = getStats(f + '/')
        allValid.append(tmpValid)
        allOracle.append(tmpOracle)
        # print(tmpValid, tmpOracle)
    saveKeys = ['mcb', 'fitz', 'cologne', '385']
    

    
    print('\n\n\n\n')
    print('VALID')
    print(methods)
    for elm in saveKeys:
        print('----')
        print(elm)
        for indx, method in enumerate(methods):
            # print('!', method)
            # print(method, allValid[indx][elm][0])
            # print(allValid[indx][elm])
            # print(allValid[indx])
            # print(method, allValid[indx][elm][0], allValid[indx][elm][1])
            print(method + f'\t{allValid[indx][elm][0]:.4f}' + f'\t{allValid[indx][elm][1]:.4f}')
    print('\n\nORACLE')
    for elm in saveKeys:
        print('----')
        print(elm)
        for indx, method in enumerate(methods):
            # print(method, allValid[indx][elm][0], allOracle[indx][elm][1])
            print(method + f'\t{allOracle[indx][elm][0]:.4f}' + f'\t{allOracle[indx][elm][1]:.4f}')
    
    print('\n---LaTeX Table---\n')
    resStr = ''
    for indx, method in enumerate(methods):
        resStr += '\n' + method
        for elm in saveKeys:
            if elm == '385':
                resStr += f' & ${allValid[indx][elm][0]:.1f} \pm {allValid[indx][elm][1]:.1f}$'
        for elm in saveKeys:
            if elm == '385':
                resStr += f' & ${allOracle[indx][elm][0]:.1f} \pm {allOracle[indx][elm][1]:.1f}$'
        resStr += '\\\\'
    print(resStr)
        

printAllStats()
# getStats('densenet121/') 
            
        
        
        
