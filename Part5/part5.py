# Step 1 understand cRF
#https://medium.com/ml2vec/overview-of-conditional-random-fields-68a2a20fa541


# Step 2 implement CRF with reference to Seong-Jin Kim 's Linear Chain CRF 

import argparse
from feature import FeatureSet, StartingIndex
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import json

#global variables
ITERATION_NUM = 0
SUB_ITERATION_NUM = 0
TOTAL_SUB_ITERATIONS = 0
GRADIENT = None
VARIANCE = 10.0

#Function used to read training file
def readTrainFile(filename):
    data = list()
    segmentLine = list(open(filename,encoding='UTF-8',mode='r'))
    word = list()
    tag = list()
    for data_string in segmentLine:
        words = data_string.strip().split()
        if len(words) is 0:
            #seperate sentenses
            data.append((word, tag))
            word = list()
            tag = list()
        else:
            word.append(words[:-1])
            tag.append(words[-1])
    if len(word) > 0:
        data.append((word, tag))
    return data
#Function used to read test file
def readTestFile(filename):
    data = list()
    segmentLine = list(open(filename,encoding='utf-8',mode='r'))
    word = list()
    for data_string in segmentLine:
        words = data_string.strip().split()
        if len(words) is 0:
            data.append((word))
            word = list()
        else:
            word.append(words)
    if len(word) > 0:
        data.append((word))

    return data

class CRF():
    def __init__(self):
        self.training_data = None
        self.feature_set = None
        self.tagSet = None
        self.tagArray = None
        self.labelCounts = None
        self.params = None
    

    def getFeature(self):
        return [[self.feature_set.get_feature_list(word, t) for t in range(len(word))]
                for word, _ in self.training_data]

    def estimation(self):

        training_feature_data = self.getFeature()
        print('Start L-BFGS-B')
        # Minimize a function func using the L-BFGS-B algorithm.
        # magic
        self.params, log_likelihood, information = fmin_l_bfgs_b(func=_log_likelihood, fprime=_gradient,x0=np.zeros(len(self.feature_set)),args=(self.training_data, self.feature_set, training_feature_data,
                                    self.feature_set.get_empirical_counts(),
                                    self.tagSet, VARIANCE),
                              callback=_callback)
        print('Finish!')
        

    #Main function to train model
    def train(self, inputFilename, model_filename):
        # Read the training corpus
        print("Reading data")
        self.training_data = readTrainFile(inputFilename)
        print("Reading Finish!")

        # Generate feature set from the input files
        self.feature_set = FeatureSet()
        self.feature_set.scan(self.training_data)
        self.tagSet, self.tagArray = self.feature_set.get_labels()
        self.labelCounts = len(self.tagArray)
        print("Number of labels:",self.labelCounts-1)
        print("Number of features: ",len(self.feature_set))

        # Estimates params to maximize log-likelihood of the input file
        self.estimation()
        self.save_model(model_filename)

        print('Training Finished' )

    #Main function to test the input file
    def test(self, test_inputFilename):
        test_data = readTestFile(test_inputFilename)
        with open ('dev.p5.out',encoding='UTF-8',mode='w') as f:
            for word in test_data:
                layer=len(word)
                Yprime = self.inference(word,layer)
                #Write the predicted result to the file
                for t in range(len(word)):
                    f.write(word[t][0])
                    f.write(' ')
                    f.write(Yprime[t])
                    f.write('\n')
                f.write('\n')    

    def save_model(self, model_filename):
        #model saved as CRF++ format
        model = {"feature_dic": self.feature_set.serialize_feature_dic(),
                 "num_features": self.feature_set.num_features,
                 "labels": self.feature_set.tagArray,
                 "params": list(self.params)}
        f = open(model_filename,encoding='utf-8',mode= 'w')
        json.dump(model, f, ensure_ascii=False, indent=2, separators=(',', ':'))
        f.close()
        print('Trained Model saved')
        
    def inference(self, word,layer):
        potential_table = generate_potential_table(self.params, self.labelCounts,self.feature_set, word, inference=True)
        #Use viterbi algorithm to decode the possible sequences
        Yprime = self.viterbi(word, potential_table,layer)
        return Yprime
 
    def viterbi(self, word, potential_table,layer): 
        pi = np.zeros((layer, self.labelCounts))
        yTable = np.zeros((layer, self.labelCounts), dtype='int64')
        #Initialization
        t = 0
        for index in range(self.labelCounts):
            pi[t, index] = potential_table[t][StartingIndex, index]
        #Recursive
        for t in range(1, layer):
            for index in range(1, self.labelCounts):
                maxPi = 0
                bestLabel = 0
                for prev_label_id in range(1, self.labelCounts):
                    value = pi[t-1, prev_label_id] * potential_table[t][prev_label_id, index]
                    if value > maxPi:
                        maxPi = value
                        bestLabel = prev_label_id
                pi[t, index] = maxPi
                yTable[t, index] = bestLabel
        sequence = list()
        next_label = pi[layer-1].argmax()
        sequence.append(next_label)
        #Final Decode
        for t in range(layer-1, -1, -1):
            next_label = yTable[t, next_label]
            sequence.append(next_label)
    
        return [self.tagSet[index] for index in sequence[::-1][1:]]

    def save_model(self, model_filename):
        #model saved as CRF++ format
        model = {"feature_dic": self.feature_set.serialize_feature_dic(),
                 "num_features": self.feature_set.num_features,
                 "labels": self.feature_set.tagArray,
                 "params": list(self.params)}
        f = open(model_filename,encoding='utf-8',mode= 'w')
        json.dump(model, f, ensure_ascii=False, indent=2, separators=(',', ':'))
        f.close()
        print('Trained CRF Model has been saved')
        
    def load(self, model_filename):
        f = open(model_filename,encoding='utf-8',mode='r')
        model = json.load(f)
        f.close()
        self.feature_set = FeatureSet()
        self.feature_set.load(model['feature_dic'], model['num_features'], model['labels'])
        self.tagSet, self.tagArray = self.feature_set.get_labels()
        self.labelCounts = len(self.tagArray)
        self.params = np.array(model['params'])

        print('CRF model loaded')

#Functions that are used in fmin_l_bfgs_b
def _callback(params):
    global ITERATION_NUM
    global SUB_ITERATION_NUM
    global TOTAL_SUB_ITERATIONS
    ITERATION_NUM += 1
    TOTAL_SUB_ITERATIONS += SUB_ITERATION_NUM
    SUB_ITERATION_NUM = 0

def generate_potential_table(params, labelCounts, feature_set, word, inference=True):

    potentialTable = list()
    for t in range(len(word)):
        table = np.zeros((labelCounts, labelCounts))
        if inference:
            for (lastY, y), score in feature_set.calc_inner_products(params, word, t):
                if lastY == -1:
                    table[:, y] += score
                else:
                    table[lastY, y] += score
        else:
            for (lastY, y), feature_ids in word[t]:
                score = sum(params[fid] for fid in feature_ids)
                if lastY == -1:
                    table[:, y] += score
                else:
                    table[lastY, y] += score
        table = np.exp(table)
        if t == 0:
            table[StartingIndex+1:] = 0
        else:
            table[:,StartingIndex] = 0
            table[StartingIndex,:] = 0
        potentialTable.append(table)

    return potentialTable

def forward_backward(labelCounts, layer, potential_table):
    #initialization
    alpha = np.zeros((layer, labelCounts))
    scaling_dic = dict()
    for index in range(labelCounts):
        alpha[0, index] = potential_table[0][StartingIndex, index]
    t = 1
    while t < layer:
        scaling_time = None
        scaling_coefficient = None
        overflow_occured = False
        index = 1
        while index < labelCounts:
            alpha[t, index] = np.dot(alpha[t-1,:], potential_table[t][:,index])
            if alpha[t, index] > 1e250:
                overflow_occured = True
                scaling_time = t - 1
                scaling_coefficient = 1e250
                scaling_dic[scaling_time] = scaling_coefficient
                break
            else:
                index += 1
        if overflow_occured:
            alpha[t-1] /= scaling_coefficient
            alpha[t] = 0
        else:
            t += 1

    beta = np.zeros((layer, labelCounts))
    for index in range(labelCounts):
        beta[layer - 1, index] = 1.0

    for i in range(layer-2, -1, -1):
        for index in range(1, labelCounts):
            beta[i, index] = np.dot(beta[i+1,:], potential_table[i+1][index,:])
        if i in scaling_dic.keys():
            beta[i] /= scaling_dic[i]

    Z = sum(alpha[layer-1])

    return alpha, beta, Z,scaling_dic

def _calc_path_score(potential_table, scaling_dic, tag, tagSet):
    score = 1.0
    lastY = StartingIndex
    for t in range(len(tag)):
        y = tagSet[tag[t]]
        score *= potential_table[lastY, y, t]
        if t in scaling_dic.keys():
            score = score / scaling_dic[t]
        lastY = y
    return score


def _log_likelihood(params,training_data, feature_set, training_feature_data, empirical_counts, tagSet, VARIANCE):

    expected_counts = np.zeros(len(feature_set))
    total_logZ = 0
    for X_features in training_feature_data:
        potential_table = generate_potential_table(params, len(tagSet), feature_set,X_features, inference=False)
        # Calculates alpha(forward terms), beta(backward terms)
        alpha, beta,Z, scaling_dic = forward_backward(len(tagSet), len(X_features), potential_table)
        total_logZ += np.log(Z) + sum(np.log(scaling_coefficient) for _, scaling_coefficient in scaling_dic.items())
        for t in range(len(X_features)):
            potential = potential_table[t]
            for (lastY, y), feature_ids in X_features[t]:
                if lastY == -1:
                    if t in scaling_dic.keys():
                        prob= (alpha[t, y] * beta[t, y] * scaling_dic[t])/Z
                    else:
                        prob= (alpha[t, y] * beta[t, y])/Z
                elif t == 0:
                    if lastY is not StartingIndex:
                        continue
                    else:
                        prob = (potential[StartingIndex, y] * beta[t, y])/Z
                else:
                    if lastY is StartingIndex or y is StartingIndex:
                        continue
                    else:
                        prob = (alpha[t-1, lastY] * potential[lastY, y] * beta[t, y])/Z
                for fid in feature_ids:
                    expected_counts[fid] += prob

    likelihood = np.dot(empirical_counts, params) - total_logZ - np.sum(np.dot(params,params))/20

    gradients = empirical_counts - expected_counts - params/10
    global GRADIENT
    GRADIENT = gradients

    print( likelihood * -1)
    return likelihood * -1

def _gradient(params, *args):
    return GRADIENT * -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str,dest="datafile", help="data file for training input")
    parser.add_argument('-m', type=str,dest="modelfile", help="the model file name. ")
    parser.add_argument('-s', type=str,dest="function", help="function selected: test or train")

    args = parser.parse_args()
    if args.function == 'test':
        crf = CRF()
        crf.load(args.modelfile)
        crf.test(args.datafile)
    
    if args.function == 'train':
        crf = CRF()
        crf.train(args.datafile, args.modelfile)
