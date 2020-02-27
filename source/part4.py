import argparse
import tqdm


def dataProcessing(filePath):
    tags = {}  
    words = {} 
    labelWords = {}  
    
    for line in open(filePath, encoding='utf-8', mode='r'):
        segmentedLine = line.rstrip()
        if segmentedLine:  
            segmentedLine = segmentedLine.rsplit(' ', 1) #x
            word = segmentedLine[0]  #y
            tag = segmentedLine[1] 
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
            if word == '#UNK#': continue
            if word not in words:
                labelWords[tag]['#UNK#'] += labelWords[tag].pop(word)
            elif words[word] < k:  
                labelWords[tag]['#UNK#'] += labelWords[tag].pop(word)
                del words[word]
            else:
                emissionPrbability[tag][word] = float(labelWords[tag][word]) / tags[tag]
        emissionPrbability[tag]['#UNK#'] = float(labelWords[tag]['#UNK#']) / tags[tag]
    trainFile= list(words)
    return trainFile,emissionPrbability

def estimateTransition(filePath):
    tags = {} 
    transitionTag = {}  
    transitionProbability = {}
    previousState = ''
    currentState = '##START##'
    for line in open(filePath, encoding='utf-8', mode='r'):
        previousState = currentState if (currentState != '##STOP##') else '##START##'  # y(i-1)
        segmentedLine = line.rstrip()

        if segmentedLine: 
            segmentedLine = segmentedLine.rsplit(' ', 1)
            currentState = segmentedLine[1]  
        else:  
            if previousState == '##START##': break  
            currentState = '##STOP##'  
        if previousState not in tags:  
            tags[previousState] = 1
            transitionTag[previousState] = {currentState: 1}
        else:
            tags[previousState] += 1
            if currentState not in transitionTag[previousState]:
                transitionTag[previousState][currentState] = 1
            else:
                transitionTag[previousState][currentState] += 1
    for tag in transitionTag:
        transitionProbability[tag] = {}
        for transition in transitionTag[tag]:
            transitionProbability[tag][transition] = float(transitionTag[tag][transition]) / tags[tag]

    return transitionProbability
    
def sentimentAnalysis(k, inputPath, m_training, emissionEstimates, transitionEstimates, outputPath):
    """ splits test file into separate observation sequences and feeds them into Viterbi algorithm """

    f = open(outputPath,encoding='utf-8',mode= 'w')

    observationSequence = []
    for line in open(inputPath, encoding='utf-8',mode='r'):
        observation = line.rstrip()
        if observation:
            observationSequence.append(observation)
        else:
            predictionSequence = kBestViterbi(k,
                                              observationSequence, m_training, emissionEstimates, transitionEstimates)
            for i in range(len(observationSequence)):
                if predictionSequence:
                    f.write('%s %s\n' %
                            (observationSequence[i], predictionSequence[i]))
                else:  # for those rare cases where the final probability is all 0
                    f.write('%s O\n' % observationSequence[i])

            f.write('\n')
            observationSequence = []

    print ('Finished writing to file %s' % (outputPath))
    return f.close()


def recursive(k, prev_layer, word, tags, m_training, emissionEstimates, transitionEstimates):
    layer = {tag: [] for tag in tags}
    # layer = {tag: [[score, word, parent_order],...[score, word, parent_order]]}
    for c_tag in tags:
        temp_scores = []
        for p_tag in tags:
            if c_tag not in transitionEstimates[p_tag]:
                continue  # only compare p_tags which can transition to c_tag

            if word in m_training:  # if this word is not #UNK#
                # and this emission can be found
                if word in emissionEstimates[c_tag]:
                    emission = emissionEstimates[c_tag][word]
                else:  # but this emission doesn't exist
                    emission = 0.00000000001
            else:  # if this word is #UNK#
                emission = emissionEstimates[c_tag]['#UNK#']

            # n scores for each prev_node
            for order in range(0, len(prev_layer[p_tag])):
                # score = prev_layer*a*b
                temp_score = prev_layer[p_tag][order][0] * \
                    transitionEstimates[p_tag][c_tag] * emission
                # 7*n scores with their parents
                temp_scores.append([temp_score, p_tag, order])
        # sort by temp_score
        temp_scores.sort(reverse=True)

        kbests = min(k, len(temp_scores))
        for kbest in range(kbests):   # get top k best
            layer[c_tag].append(temp_scores[kbest])

    return layer


def kBestViterbi(k, observationSequence, m_training, emissionEstimates, transitionEstimates):
    """ K Best Viterbi algorithm """
    tags = list(emissionEstimates)
    pi = [{tag: []
           for tag in list(emissionEstimates)} for o in observationSequence]
    # pi = [{layer0}, {layer2}, ..., {layerN}] where 0->START and N+1->STOP
    # layer = {tag: [[score, parent_word, parent_order],...[score, parent_word, parent_order]]}

    # Initialization
    for c_tag in tags:
        if c_tag not in transitionEstimates['##START##']:
            continue  # update tags which can be transitioned from ##START##

        if observationSequence[0] in m_training:  # if this word is not #UNK#
            # and this emission can be found
            if observationSequence[0] in emissionEstimates[c_tag]:
                emission = emissionEstimates[c_tag][observationSequence[0]]
            else:  # but this emission doesn't exist
                emission = 0.00000000001
        else:  # if this word is #UNK#
            emission = emissionEstimates[c_tag]['#UNK#']

        pi[0][c_tag].append([transitionEstimates['##START##']
                             [c_tag] * emission, '##START##', 0])

    # Recursive case
    for o in tqdm.tqdm(range(1, len(observationSequence))):
        word = observationSequence[o]
        prev_layer = pi[o-1]
        curr_layer = recursive(k, prev_layer, word, tags,
                               m_training, emissionEstimates, transitionEstimates)
        pi[o] = curr_layer

    # Finally
    result = []
    temp_scores = []
    for p_tag in tags:
        if '##STOP##' not in transitionEstimates[p_tag]:
            continue  # only compare p_tags which can transition to ##STOP##
        # n scores for each prev_node
        for order in range(0, len(pi[-1][p_tag])):
            # score = prev_layer*a*b
            temp_score = pi[-1][p_tag][order][0] * \
                transitionEstimates[p_tag]['##STOP##']
            # 7*n scores with their parents
            temp_scores.append([temp_score, p_tag, order])
    # sort by temp_score
    temp_scores.sort(key=lambda tup: tup[0], reverse=True)

    kbests = min(k, len(temp_scores))
    for kbest in range(kbests):   # get top k best
        result.append(temp_scores[kbest])

    # Backtracking
    parent_order = min(k-1, len(result)) - 1
    if not result[parent_order][1]:  # for those weird cases where the final probability is 0
        return

    prediction = [result[parent_order][1]]
    parent_order = result[parent_order][2]

    for o in reversed(range(len(observationSequence))):
        if o == 0:
            break  # skip ##START## tag
        # print (parent_order, len(pi[o][prediction[0]]))
        prediction.insert(0, pi[o][prediction[0]][parent_order][1])
        parent_order = pi[o][prediction[1]][parent_order][2]

    return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, dest='dataset',
                        help='Dataset to run script over', required=True)

    args = parser.parse_args()

    trainFilePath = '../%s/train' % (args.dataset)
    inputTestFilePath = '../%s/dev.in' % (args.dataset)
    outputTestFilePath = '../%s/dev.p4.out' % (args.dataset)

    transitionEstimates = estimateTransition(trainFilePath)
    tags,words,labelWords= dataProcessing(trainFilePath)
    trainFile,emissionEstimates = smoothedEstimation(tags,words,labelWords,k=3)
    sentimentAnalysis(7, inputTestFilePath, trainFile,
                      emissionEstimates, transitionEstimates, outputTestFilePath)
