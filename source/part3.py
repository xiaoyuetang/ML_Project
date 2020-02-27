import argparse
import tqdm

def lineCut(segmentedLine):
    segmentedLine = segmentedLine.rsplit(' ', 1)
    word = segmentedLine[0] 
    tag = segmentedLine[1] 
    return word,tag

def dataProcessing(filePath):
    tags = {}  
    words = {} 
    labelWords = {}  
    
    for line in open(filePath, encoding='utf-8', mode='r'):
        segmentedLine = line.strip()
        if segmentedLine:  
            word,tag=lineCut(segmentedLine)
            if word not in words:  
                words[word] = 1
            else:  
                words[word] += 1
            if tag not in tags:  
                tags[tag] = 1
                labelWords[tag] = {word: 1}
            else: 
                tags[tag] += 1
                if word not in labelWords[tag]:
                    labelWords[tag][word] = 1
                else:
                    labelWords[tag][word] += 1

    return tags,words,labelWords

def smoothedEstimation(tags,words,labelWords,k):
    emissionPrbability = {}
    for tag in labelWords:
        emissionPrbability[tag] = {}
        labelWords[tag]['#UNK#'] = 0
        for word in list(labelWords[tag]): 
            if word not in words:
                labelWords[tag]['#UNK#'] += labelWords[tag].pop(word)
            elif words[word] < k:  
                labelWords[tag]['#UNK#'] += labelWords[tag].pop(word)
                del words[word]
            else:
                emissionPrbability[tag][word] = labelWords[tag][word] / tags[tag]
        emissionPrbability[tag]['#UNK#'] = labelWords[tag]['#UNK#'] / tags[tag]
    trainFile= list(words)
    return trainFile,emissionPrbability


def estimateTransition(filePath):
    tags = {} 
    transitionTag = {}  
    transitionProbability = {}
    _preT = ''
    _newT = '##START##'
    for line in open(filePath, encoding='utf-8', mode='r'):
        _preT = _newT if (_newT != '##STOP##') else '##START##' 
        segmentedLine = line.rstrip()

        if segmentedLine: 
            segmentedLine = segmentedLine.rsplit(' ', 1)
            _newT = segmentedLine[1]  
        else:  
            _newT = '##STOP##'  
        if _preT not in tags:  
            tags[_preT] = 1
            transitionTag[_preT] = {_newT: 1}
        else:
            tags[_preT] += 1
            if _newT not in transitionTag[_preT]:
                transitionTag[_preT][_newT] = 1
            else:
                transitionTag[_preT][_newT] += 1

    calculateTransitionProbability(transitionProbability,transitionTag,tags)

    return transitionProbability


def calculateTransitionProbability(transitionProbability,transitionTag,tags):
    for tag in transitionTag:
        transitionProbability[tag] = {}
        for transition in transitionTag[tag]:
            transitionProbability[tag][transition] = transitionTag[tag][transition]/ tags[tag]


def writeToFile(inputPath, trainingSet, emissionEstimates, transitionEstimates, outputPath):
    f = open(outputPath,encoding='utf-8', mode= 'w')
    sequence = []
    for line in open(inputPath, encoding='utf-8', mode='r'):
        word = line.rstrip()
        if word:
            sequence.append(word)
        else:
            predictionSequence = viterbi(sequence, trainingSet, emissionEstimates, transitionEstimates)
            for i in range(len(sequence)):
                if predictionSequence:
                    f.write('{0} {1}\n'.format(sequence[i], predictionSequence[i]))

                else: 
                    f.write('{0} O\n' .format(sequence[i]))

            f.write('\n')
            sequence = []

    print ('Finished writing to file')
    return f.close()

def getEstimate(sequence,trainingSet,label,k=0):
    if sequence[k] in trainingSet:  
        if sequence[k] in emissionEstimates[label]:  
            emission = emissionEstimates[label][sequence[k]]
        else:  
            emission = 0.0
    else:  
        emission = emissionEstimates[label]['#UNK#']   
    return emission
def viterbi(sequence, trainingSet, emissionEstimates, transitionEstimates):
    tags = list(emissionEstimates)
    
    pi = [{tag: [0.0, ''] for tag in tags} for o in sequence]

    # Initialization  stage
    for label in tags :
        if label not in transitionEstimates['##START##']: continue
        emission=getEstimate(sequence,trainingSet,label)
   

        pi[0][label] = [transitionEstimates['##START##'][label] * emission]

    for k in tqdm.tqdm(range(1, len(sequence))):  
        for label in tags:
            piList=[]
            for transTag in tags:
                if label not in transitionEstimates[transTag]: continue  
                score = pi[k-1][transTag][0] * transitionEstimates[transTag][label]
                piList.append([score, transTag])
            piList.sort(reverse=True)
            pi[k][label]=piList[0]

            if sequence[k] in trainingSet: 
                if sequence[k] in emissionEstimates[label]:  
                    emission = emissionEstimates[label][sequence[k]]
                else:  
                    emission = 0.0
            else:  
                emission = emissionEstimates[label]['#UNK#']
            pi[k][label][0] *= emission

    # Finally
    slist=[]
    result = [0.0, '']
    for transTag in tags:
        if '##STOP##' not in transitionEstimates[transTag]: continue  
        score = pi[-1][transTag][0] * transitionEstimates[transTag]['##STOP##']
        slist.append([score, transTag])
    slist.sort(reverse=True)
    result=slist[0]
    # Backtracking
    if not result[1]:  
        return

    prediction = [result[1]]
    for k in reversed(range(len(sequence))):
        if k == 0: break  
        prediction.insert(0, pi[k][prediction[0]][1])

    return prediction

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, dest='dataset', help='Dataset to run script over', required=True)

    args = parser.parse_args()

    trainFilePath = './%s/train' % (args.dataset)
    inputTestFilePath = './%s/dev.in' % (args.dataset)
    outputTestFilePath = './%s/dev.p3.out' % (args.dataset)

    transitionEstimates = estimateTransition(trainFilePath)
    tags,words,labelWords= dataProcessing(trainFilePath)
    trainFile,emissionEstimates = smoothedEstimation(tags,words,labelWords,k=3)
    writeToFile(inputTestFilePath, trainFile, emissionEstimates, transitionEstimates, outputTestFilePath)
