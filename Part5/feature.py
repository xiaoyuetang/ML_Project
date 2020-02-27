
from collections import Counter
import numpy as np


STARTING_LABEL = '*'        # Label of t=-1
STARTING_LABEL_INDEX = 0


def default_feature_func(_, word, t):
    #Default feature list for project data format
    length = len(word)

    features = list()
    features.append('U[0]:%s' % word[t][0])
    if t < length-1:
        features.append('U[+1]:%s' % (word[t+1][0]))
        features.append('B[0]:%s %s' % (word[t][0], word[t+1][0]))
        if t < length-2:
            features.append('U[+2]:%s' % (word[t+2][0]))
    if t > 0:
        features.append('U[-1]:%s' % (word[t-1][0]))
        features.append('B[-1]:%s %s' % (word[t-1][0], word[t][0]))
        if t > 1:
            features.append('U[-2]:%s' % (word[t-2][0]))


    return features


class FeatureSet():
    feature_dic = dict()
    observation_set = set()
    empirical_counts = Counter()
    num_features = 0
    tagSet = {STARTING_LABEL: STARTING_LABEL_INDEX}
    tagArray = [STARTING_LABEL]

    feature_func = default_feature_func

    def __init__(self, feature_func=None):
        # Sets a custom feature function.
        if feature_func is not None:
            self.feature_func = feature_func

    def scan(self, data):
        # data= (word,tag)
        # Constructs a feature set, and counts empirical counts.
        for word, tag in data:
            lastY = STARTING_LABEL_INDEX
            for t in range(len(word)):
                # Gets a label id
                try:
                    y = self.tagSet[tag[t]]
                except KeyError:
                    y = len(self.tagSet)
                    self.tagSet[tag[t]] = y
                    self.tagArray.append(tag[t])
                # Adds features
                self._add(lastY, y, word, t)
                lastY = y

    def load(self, feature_dic, num_features, tagArray):
        self.num_features = num_features
        self.tagArray = tagArray
        self.tagSet = {label: i for label, i in enumerate(tagArray)}
        self.feature_dic = self.deserialize_feature_dic(feature_dic)

    def __len__(self):
        return self.num_features

    def _add(self, lastY, y, word, t):
   
        for feature_string in self.feature_func(word, t):
            if feature_string in self.feature_dic.keys():
                if (lastY, y) in self.feature_dic[feature_string].keys():
                    self.empirical_counts[self.feature_dic[feature_string][(lastY, y)]] += 1
                else:
                    feature_id = self.num_features
                    self.feature_dic[feature_string][(lastY, y)] = feature_id
                    self.empirical_counts[feature_id] += 1
                    self.num_features += 1
                if (-1, y) in self.feature_dic[feature_string].keys():
                    self.empirical_counts[self.feature_dic[feature_string][(-1, y)]] += 1
                else:
                    feature_id = self.num_features
                    self.feature_dic[feature_string][(-1, y)] = feature_id
                    self.empirical_counts[feature_id] += 1
                    self.num_features += 1
            else:
                self.feature_dic[feature_string] = dict()
                # Bigram feature
                feature_id = self.num_features
                self.feature_dic[feature_string][(lastY, y)] = feature_id
                self.empirical_counts[feature_id] += 1
                self.num_features += 1
                # Unigram feature
                feature_id = self.num_features
                self.feature_dic[feature_string][(-1, y)] = feature_id
                self.empirical_counts[feature_id] += 1
                self.num_features += 1

    def get_feature_vector(self, lastY, y, word, t):
    
        feature_ids = list()
        for feature_string in self.feature_func(word, t):
            try:
                feature_ids.append(self.feature_dic[feature_string][(lastY, y)])
            except KeyError:
                pass
        return feature_ids

    def get_labels(self):

        return self.tagSet, self.tagArray

    def calc_inner_products(self, params, word, t):

        inner_products = Counter()
        for feature_string in self.feature_func(word, t):
            try:
                for (lastY, y), feature_id in self.feature_dic[feature_string].items():
                    inner_products[(lastY, y)] += params[feature_id]
            except KeyError:
                pass
        return [((lastY, y), score) for (lastY, y), score in inner_products.items()]

    def get_empirical_counts(self):
        empirical_counts = np.ndarray((self.num_features,))
        for feature_id, counts in self.empirical_counts.items():
            empirical_counts[feature_id] = counts
        return empirical_counts

    def get_feature_list(self, word, t):
        feature_list_dic = dict()
        for feature_string in self.feature_func(word, t):
            for (lastY, y), feature_id in self.feature_dic[feature_string].items():
                if (lastY, y) in feature_list_dic.keys():
                    feature_list_dic[(lastY, y)].add(feature_id)
                else:
                    feature_list_dic[(lastY, y)] = {feature_id}
        return [((lastY, y), feature_ids) for (lastY, y), feature_ids in feature_list_dic.items()]

    def serialize_feature_dic(self):
        serialized = dict()
        for feature_string in self.feature_dic.keys():
            serialized[feature_string] = dict()
            for (lastY, y), feature_id in self.feature_dic[feature_string].items():
                serialized[feature_string]['%d_%d' % (lastY, y)] = feature_id
        return serialized

    def deserialize_feature_dic(self, serialized):
        feature_dic = dict()
        for feature_string in serialized.keys():
            feature_dic[feature_string] = dict()
            for transition_string, feature_id in serialized[feature_string].items():
                lastY, y = transition_string.split('_')
                feature_dic[feature_string][(int(lastY), int(y))] = feature_id
        return feature_dic