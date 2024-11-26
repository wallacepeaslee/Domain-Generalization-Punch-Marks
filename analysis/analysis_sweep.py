# -*- coding: utf-8 -*-


import json
import os
import numpy as np

ENVIRONMENT_LENGTHS = []

#492, 98/394
#2211, 442/1769
#712, 142/670
EVALUATION_WEIGHTS = [98,442,142]
EVALUATION_WEIGHTS = [x/sum(EVALUATION_WEIGHTS) for x in EVALUATION_WEIGHTS]

# ENV0_WEIGHTS = [1,1]
ENV_SIZE = {'env0_in': 289,
            'env0_out': 72,
            'env1_in': 59,
            'env1_out': 14,
            'env2_in': 11,
            'env2_out': 2,
            'env3_in': 36,
            'env3_out': 8}



def getResultData(fp):
    '''
    given filepath of .jsonl file
    returns a list, where each element is dict of checkpoint
    '''
    ret = []
    with open(fp, 'r', encoding='utf-8') as j:
        for line in j:
            ret.append(json.loads(line)) 
    return ret

def bestValidationModel(d, weighted = False):
    '''

    Parameters
    ----------
    d : list of dictionaries giving training data stuff.

    list index, acc, dictionary of results where best valid model is

    '''
    if weighted:
        w = EVALUATION_WEIGHTS
    else:
        w = [1/3, 1/3, 1/3]
    
    currBestI = 0
    currBestA = 0
    for indx, cp in enumerate(d):
        currAcc = 0
        currAcc += cp['env4_out_acc']*w[0]
        currAcc += cp['env5_out_acc']*w[1]
        currAcc += cp['env6_out_acc']*w[2]
        if currAcc > currBestA:
            currBestA = currAcc
            currBestI = indx
    return currBestI, currBestA, d[currBestI]

def getEnvAccuracy(d, num):
    totalEnvNum = ENV_SIZE['env' + str(num) + '_in'] + ENV_SIZE['env' + str(num) + '_out']
    return d['env' + str(num) + '_in_acc']*ENV_SIZE['env' + str(num) + '_in']/totalEnvNum + d['env' + str(num) + '_out_acc']*ENV_SIZE['env' + str(num) + '_out']/totalEnvNum


def getEnvOracle(d, num):
    totalEnvNum = ENV_SIZE['env' + str(num) + '_in'] + ENV_SIZE['env' + str(num) + '_out']
    return d[-1]['env' + str(num) + '_in_acc']*ENV_SIZE['env' + str(num) + '_in']/totalEnvNum + d[-1]['env' + str(num) + '_out_acc']*ENV_SIZE['env' + str(num) + '_out']/totalEnvNum

def Stats(d):
    data = np.array(d)
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    return mean, err


rootdir = 'data' #TODO: update root directory
experiments = os.listdir(rootdir)
data = {}
for exp in experiments:
    exppath = os.path.join(rootdir, exp)
    jsonlfile = os.path.join(exppath, 'results.jsonl')
    jsonData = getResultData(jsonlfile)
    info = exp.strip().split("_")
    alg = info[1]
    largeTrial = int(info[3][1:])
    subTrial = int(info[2][1:])
    if alg not in data:
        data[alg] = {}
    if largeTrial not in data[alg]:
        data[alg][largeTrial] = {}
    data[alg][largeTrial][subTrial] = jsonData
    

    


algs = sorted(data.keys())


r1 = ''
r2 = ''
d1 = {}
d2 = {}

'''
MCB
'''

print('\n MCB \n')


print('\n --- Unweighted Train Validation --- MCB')

res = {}
resDet = {}
for a in algs:
    res[a] = []
    trials = sorted(data[a].keys())
    resDet[a] = {}
    for t in trials:
        bestAcc = 0
        bestIndx = -1
        currData = None
        bestCheckPoint = -1
        for sub in sorted(data[a][t].keys()):
            staData = data[a][t][sub]
            checkPoint, acc, subTrialData = bestValidationModel(staData)
            #can replace above with other model selection method
            if acc > bestAcc:
                bestAcc = acc
                bestIndx = sub
                currData = subTrialData
                bestCheckPoint = checkPoint
        bestTrialAccuracy = getEnvAccuracy(currData, 0)
        res[a].append(bestTrialAccuracy)
        resDet[a][t] = (bestIndx, currData['step'])
        
for a in sorted(res.keys()):
    mean, se = Stats(res[a])
    print(a, mean, se)
    


print('\n --- Weighted Train Validation --- MCB')

res = {}
resDet = {}
for a in algs:
    res[a] = []
    trials = sorted(data[a].keys())
    resDet[a] = {}
    for t in trials:
        bestAcc = 0
        bestIndx = -1
        currData = None
        bestCheckPoint = -1
        for sub in sorted(data[a][t].keys()):
            staData = data[a][t][sub]
            checkPoint, acc, subTrialData = bestValidationModel(staData, weighted=True)
            #can replace above with other model selection method
            if acc > bestAcc:
                bestAcc = acc
                bestIndx = sub
                currData = subTrialData
                bestCheckPoint = checkPoint
        bestTrialAccuracy = getEnvAccuracy(currData, 0)
        res[a].append(bestTrialAccuracy)
        resDet[a][t] = (bestIndx, currData['step'])
        
for a in sorted(res.keys()):
    mean, se = Stats(res[a])
    print(a, mean, se)
    d1[a] = {}
    d1[a]['MCB'] = (mean, se)
 

print('\n --- Oracle Validation --- MCB')

res = {}
resDet = {}
for a in algs:
    res[a] = []
    trials = sorted(data[a].keys())
    resDet[a] = {}
    for t in trials:
        bestAcc = 0
        bestIndx = -1
        currData = None
        bestCheckPoint = -1
        for sub in sorted(data[a][t].keys()):
            staData = data[a][t][sub]
            acc = getEnvOracle(staData, 0)
            checkPoint = -1
            subTrialData = staData[-1]
            #can replace above with other model selection method
            if acc > bestAcc:
                bestAcc = acc
                bestIndx = sub
                currData = subTrialData
                bestCheckPoint = checkPoint
        bestTrialAccuracy = getEnvAccuracy(currData, 0)
        res[a].append(bestTrialAccuracy)
        resDet[a][t] = (bestIndx, currData['step'])
        
        


for a in sorted(res.keys()):
    mean, se = Stats(res[a])
    print(a, mean, se)
    d2[a] = {}
    d2[a]['MCB'] = (mean, se)




'''
Fitz
'''

print('\n FITZ \n')

print('\n --- Unweighted Train Validation --- Fitz')

res = {}
resDet = {}
for a in algs:
    res[a] = []
    trials = sorted(data[a].keys())
    resDet[a] = {}
    for t in trials:
        bestAcc = 0
        bestIndx = -1
        currData = None
        bestCheckPoint = -1
        for sub in sorted(data[a][t].keys()):
            staData = data[a][t][sub]
            checkPoint, acc, subTrialData = bestValidationModel(staData)
            #can replace above with other model selection method
            if acc > bestAcc:
                bestAcc = acc
                bestIndx = sub
                currData = subTrialData
                bestCheckPoint = checkPoint
        bestTrialAccuracy = getEnvAccuracy(currData, 1)
        res[a].append(bestTrialAccuracy)
        resDet[a][t] = (bestIndx, currData['step'])
        
for a in sorted(res.keys()):
    mean, se = Stats(res[a])
    print(a, mean, se)
    
# for a in sorted(res.keys()):
#     print(a, sorted(res[a]))


# print('\n\n--Details--')
# for a in sorted(resDet.keys()):
#     print(a, resDet[a])



print('\nWeighted Train Validation --- Fitz')

res = {}
resDet = {}
for a in algs:
    res[a] = []
    trials = sorted(data[a].keys())
    resDet[a] = {}
    for t in trials:
        bestAcc = 0
        bestIndx = -1
        currData = None
        bestCheckPoint = -1
        for sub in sorted(data[a][t].keys()):
            staData = data[a][t][sub]
            checkPoint, acc, subTrialData = bestValidationModel(staData, weighted=True)
            #can replace above with other model selection method
            if acc > bestAcc:
                bestAcc = acc
                bestIndx = sub
                currData = subTrialData
                bestCheckPoint = checkPoint
        bestTrialAccuracy = getEnvAccuracy(currData, 1)
        res[a].append(bestTrialAccuracy)
        resDet[a][t] = (bestIndx, currData['step'])
        
for a in sorted(res.keys()):
    mean, se = Stats(res[a])
    print(a, mean, se)
    d1[a]['Fitz'] = (mean, se)
    
# for a in sorted(res.keys()):
#     print(a, sorted(res[a]))


# print('\n\n--Details--')
# for a in sorted(resDet.keys()):
#     print(a, resDet[a])
    


print('\nOracle Validation --- Fitz')

res = {}
resDet = {}
for a in algs:
    res[a] = []
    trials = sorted(data[a].keys())
    resDet[a] = {}
    for t in trials:
        bestAcc = 0
        bestIndx = -1
        currData = None
        bestCheckPoint = -1
        for sub in sorted(data[a][t].keys()):
            staData = data[a][t][sub]
            acc = getEnvOracle(staData, 1)
            checkPoint = -1
            subTrialData = staData[-1]
            #can replace above with other model selection method
            if acc > bestAcc:
                bestAcc = acc
                bestIndx = sub
                currData = subTrialData
                bestCheckPoint = checkPoint
        bestTrialAccuracy = getEnvAccuracy(currData, 1)
        res[a].append(bestTrialAccuracy)
        resDet[a][t] = (bestIndx, currData['step'])
        


for a in sorted(res.keys()):
    mean, se = Stats(res[a])
    print(a, mean, se)
    d2[a]['Fitz'] = (mean, se)
    


'''
Cologne
'''

print('\n COLOGNE \n')


print('\n --- Unweighted Train Validation --- Cologne')

res = {}
resDet = {}
for a in algs:
    res[a] = []
    trials = sorted(data[a].keys())
    resDet[a] = {}
    for t in trials:
        bestAcc = 0
        bestIndx = -1
        currData = None
        bestCheckPoint = -1
        for sub in sorted(data[a][t].keys()):
            staData = data[a][t][sub]
            checkPoint, acc, subTrialData = bestValidationModel(staData)
            #can replace above with other model selection method
            if acc > bestAcc:
                bestAcc = acc
                bestIndx = sub
                currData = subTrialData
                bestCheckPoint = checkPoint
        bestTrialAccuracy = getEnvAccuracy(currData, 2)
        res[a].append(bestTrialAccuracy)
        resDet[a][t] = (bestIndx, currData['step'])
        
for a in sorted(res.keys()):
    mean, se = Stats(res[a])
    print(a, mean, se)
  

print('\n --- Weighted Train Validation --- Cologne')

res = {}
resDet = {}
for a in algs:
    res[a] = []
    trials = sorted(data[a].keys())
    resDet[a] = {}
    for t in trials:
        bestAcc = 0
        bestIndx = -1
        currData = None
        bestCheckPoint = -1
        for sub in sorted(data[a][t].keys()):
            staData = data[a][t][sub]
            checkPoint, acc, subTrialData = bestValidationModel(staData, weighted=True)
            #can replace above with other model selection method
            if acc > bestAcc:
                bestAcc = acc
                bestIndx = sub
                currData = subTrialData
                bestCheckPoint = checkPoint
        bestTrialAccuracy = getEnvAccuracy(currData, 2)
        res[a].append(bestTrialAccuracy)
        resDet[a][t] = (bestIndx, currData['step'])
        
for a in sorted(res.keys()):
    mean, se = Stats(res[a])
    print(a, mean, se)
    d1[a]['Col'] = (mean, se)
    



print('\n --- Oracle Validation --- Cologne')

res = {}
resDet = {}
for a in algs:
    res[a] = []
    trials = sorted(data[a].keys())
    resDet[a] = {}
    for t in trials:
        bestAcc = -1
        bestIndx = -1
        currData = None
        bestCheckPoint = -1
        for sub in sorted(data[a][t].keys()):
            staData = data[a][t][sub]
            acc = getEnvOracle(staData, 2)
            checkPoint = -1
            subTrialData = staData[-1]
            #can replace above with other model selection method
            if acc > bestAcc:
                bestAcc = acc
                bestIndx = sub
                currData = subTrialData
                bestCheckPoint = checkPoint
        bestTrialAccuracy = getEnvAccuracy(currData, 2)
        res[a].append(bestTrialAccuracy)
        resDet[a][t] = (bestIndx, currData['step'])
        


for a in sorted(res.keys()):
    mean, se = Stats(res[a])
    print(a, mean, se)
    d2[a]['Col'] = (mean, se)

def dictToTable(d):
    # w = [400, 64, 13]
    res = ''
    for alg in sorted(d.keys()):
        res += '\n' + alg
        for domain in sorted(d[alg].keys(), reverse = True):
            res += ' ' + domain
            
    res += '\n\n'
    for alg in sorted(d.keys()):
        res += '\n' + alg
        for domain in sorted(d[alg].keys(), reverse = True):
            res += '    & ' + '${0:.1f}'.format(d[alg][domain][0]) + ' \pm {0:.1f}$ '.format(d[alg][domain][1])
        res += '\\\\'
    return res


r1 = dictToTable(d1)
r2 = dictToTable(d2)

print('\n\nValidation Selection')
print(r1)


print('\n\nOracle Selection')
print(r2)


def dictToChart(d):
    res = ['' for _ in range(7)]
    for alg in ['ERM',  'ERM-unfreeze', 'ERMNoAug', 'ERMRes18', 'ERM-unpretrained', 'IRM', 'GroupDRO', 'Mixup', 'MLDG', 'CORAL', 'MMD', 'DANN', 'CDANN', 'SagNet', 'EQRM']:
        res[0] += alg + '\n'
        res[1] += '{0:.4f}'.format(d[alg]['MCB'][0]) + '\n'
        res[2] += '{0:.4f}'.format(d[alg]['Fitz'][0]) + '\n'
        res[3] += '{0:.4f}'.format(d[alg]['Col'][0]) + '\n'
        res[4] += '{0:.4f}'.format(d[alg]['MCB'][1]) + '\t'
        res[5] += '{0:.4f}'.format(d[alg]['Fitz'][1]) + '\t'
        res[6] += '{0:.4f}'.format(d[alg]['Col'][1]) + '\t'
    return res


print("\nValidation Accuracy")
p = dictToChart(d1)
for x in p:
    print(x)

print("\nOracle Accuracy")
p = dictToChart(d2)
for x in p:
    print(x)