import argparse

def lineCut(segmentedLine):
    segmentedLine = segmentedLine.rsplit(' ', 1)
    word = segmentedLine[0] 
    tag = segmentedLine[1] 
    return word,tag

def dataProcessing(filePath):
    tags = {}  
    words = {} 
    labelWords = {}  
    estimates = {}
    for line in open(filePath, encoding='utf-8', mode='r'):
        segmentedLine = line.rstrip()
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

    return tags,words,labelWords,estimates

def smoothedEstimation(tags,words,labelWords,estimates,k):
    for tag in labelWords:
        estimates[tag] = {}
        labelWords[tag]['#UNK#'] = 0
        for word in list(labelWords[tag]):
            if word not in words or words[word] < k:  
                labelWords[tag]['#UNK#'] += labelWords[tag][word]
            else:
                estimates[tag][word] = (labelWords[tag][word]) / tags[tag]
        estimates[tag]['#UNK#'] = (labelWords[tag]['#UNK#']) / tags[tag]

    return estimates

def getBestOutput(word,estimates):
   
    prediction = ['', 0.0]
    unkPrediction = ['#UNK#', 0.0]
    for tag in estimates:
        if estimates[tag]['#UNK#'] > unkPrediction[1]:
            unkPrediction[0] = tag
            unkPrediction[1] = estimates[tag]['#UNK#']
        if word in estimates[tag] and estimates[tag][word] > prediction[1]:
            prediction[1] = estimates[tag][word]
            prediction[0] = tag
    return unkPrediction,prediction


def writeFile(inputPath, estimates, outputPath):
    f = open(outputPath,encoding='utf-8',mode= 'w')
    for line in open(inputPath,encoding='utf-8', mode= 'r'):
        word = line.rstrip()
        if word:
            unkPrediction,prediction = getBestOutput(word,estimates)
            if prediction[0]:
                f.write('{0} {1}\n'.format(word, prediction[0]))
            else:
                f.write('{0} {1}\n'.format(word, unkPrediction[0]))
        else:
            f.write('\n')

    print ('Finished writing to file')
    return f.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, dest='dataset', help='Dataset to run script over', required=True)

    args = parser.parse_args()

    trainFilePath = './%s/train' % (args.dataset)
    inputTestFilePath = './%s/dev.in' % (args.dataset)
    outputTestFilePath = './%s/dev.p2.out' % (args.dataset)

    tags,words,labelWords,estimates= dataProcessing(trainFilePath)
    estimates=smoothedEstimation(tags,words,labelWords,estimates,k=3)
    writeFile(inputTestFilePath, estimates, outputTestFilePath)