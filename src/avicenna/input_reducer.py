import numpy as np
import copy
import math

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn import decomposition
from scipy.stats import pearsonr

from fuzzingbook.Grammars import Grammar

from avicenna.features import Feature, FeatureVector
from avicenna.input import Input
from avicenna.feature_collector import (
    FeatureFactory,
    ExistenceFeature,
    DerivationFeature,
    NumericFeature,
    LengthFeature,
    GrammarFeatureCollector,
)

#from tests import test_features

def determine_input_subset(inputs, grammar, selection_size) -> list:
    input_feature_vector_list = generate_features(inputs, grammar, [ExistenceFeature, LengthFeature, NumericFeature])
    input_feature_vector_list = remove_monotone_features(input_feature_vector_list)
    normalized_input_feature_vector_list = normalize_features(input_feature_vector_list)
    reduced_input_feature_vector_list = reduce_dimensionality(normalized_input_feature_vector_list)
    selection_set = select_dissimilar_inputs(reduced_input_feature_vector_list, selection_size)


def generate_features(inputs, grammar, feature_types=[ExistenceFeature, 
                                                    DerivationFeature, 
                                                    LengthFeature, 
                                                    NumericFeature]):
    """Generates the input_feature_vector
    :param inputs: the list of inputs to be transformed in an input_feature_vector_list
    :param grammar: the grammar used to derive features
    :feature_types: a list of the selected feature types, of which features shall be generated"""
    feature_list = list()
    
    test_inputs = inputs

    #test_inputs = [Input.from_str(grammar, inp) for inp in inputs]

    collector = GrammarFeatureCollector(grammar, feature_types)
    cnt = 0
    for test_input in test_inputs:
            feature_vector = collector.collect_features(test_input)
            #print(f"feature_vector: {feature_vector}")
            feature_list.append(feature_vector)
            cnt = cnt + 1

    return feature_list


def remove_monotone_features(input_feature_vector_list: list()):
    """Removes all monotone features, meaning features that have the same value for all given inputs, from the 
    input_feature_vector"""
    feature_dict = input_feature_vector_list[0].get_features()
    features_value = {}
    monotone_features = list(feature_dict.keys())

    for feature_name in feature_dict:
        features_value[feature_name] =  feature_dict[feature_name]

    # find monotone features (where all values are the same) 
    for input_feature_vector in input_feature_vector_list:
        for feature_name in feature_dict:
            if (feature_name in monotone_features and
                features_value[feature_name] != input_feature_vector.get_features()[feature_name]):
                monotone_features.remove(feature_name)
    
    # remove all key-value pairs for the found monotone features
    for input_feature_vector in input_feature_vector_list:
        input_feature_vector.remove_features(monotone_features)
        
    return input_feature_vector_list

def remove_redundant_features(input_vector_list: list):
    feature_vectors = create_feature_dict(input_vector_list)
    redundant_features = set()
    for A, featureA in enumerate(feature_vectors):
        for B, featureB in enumerate(feature_vectors):
            if B <= A:
                continue
            if np.array_equal(feature_vectors[featureA], feature_vectors[featureB]):
                redundant_features.add(featureB)

    for input_vector in input_vector_list:
        input_vector.remove_features(redundant_features)

    return input_vector_list

def pca(input_feature_vector_list):
    feature_matrix, feature_names = create_feature_matrix(input_feature_vector_list)
    feature_matrix = np.where(np.isinf(feature_matrix), 0, feature_matrix)
    feature_matrix = np.where(np.isnan(feature_matrix), 0, feature_matrix)
    
    scaler = StandardScaler()
    scaled_feature_matrix = scaler.fit_transform(feature_matrix)


    print(f"BEFORE PCA: {np.shape(scaled_feature_matrix)}")
    pca = decomposition.PCA(n_components='mle', svd_solver='full')
    transformed_feature_matrix = pca.fit_transform(scaled_feature_matrix)
    print(f"AFTER PCA: {np.shape(transformed_feature_matrix)}")
    return transformed_feature_matrix

def reduce_dimensionality(input_feature_vector_list, with_PCA=False):
    input_feature_vector_list = copy.deepcopy(input_feature_vector_list)
    input_feature_vector_list = remove_monotone_features(input_feature_vector_list)
    input_feature_vector_list = remove_redundant_features(input_feature_vector_list)

    if with_PCA:
        input_feature_vector_list = pca(input_feature_vector_list)
    return input_feature_vector_list

def normalize_features(input_feature_vector_list):
    input_feature_vector_list = copy.deepcopy(input_feature_vector_list)
    features = input_feature_vector_list[0].get_features()
    for feature in features:
        if feature.type == bool:
            continue
        feature_values = collect_feature_values(input_feature_vector_list, feature)
        max_feature_value = np.max(feature_values)
        min_feature_value = np.min(feature_values)
        normed_feature_values = np.divide(feature_values - min_feature_value, max_feature_value - min_feature_value) 
        for i in range(normed_feature_values.size):
            input_feature_vector_list[i].features[feature] = normed_feature_values[i]
        
    return input_feature_vector_list



def calculate_distance(x: np.array, y: np.array, p: float) -> float:
    elem_distance = np.absolute(x - y)
    elem_distance_potentiated = np.power(elem_distance, p)
    elem_distance_potentiated = np.where(np.isinf(elem_distance_potentiated), 0, elem_distance_potentiated)
    elem_distance_potentiated = np.where(np.isnan(elem_distance_potentiated), 0, elem_distance_potentiated)

    summed_elem_distance = np.sum(elem_distance_potentiated)
    distance = np.power(summed_elem_distance, 1/p)
    if math.isnan(distance):
        print(f"IS NAN {x}, {y}")
    return distance

def select_dissimilar_inputs(input_feature_vector_list, selection_size, preselected_inputs=None, p_value=1):
    if len(input_feature_vector_list) < selection_size:
        raise Exception("Cannot produce selection that is bigger than base set!")
    selected_inputs = preselected_inputs
    if selected_inputs is None:
        selected_inputs = set()
    
    for _ in range(selection_size - len(selected_inputs)):
        max_dist = 0
        max_dist_index = -1
        
        if selected_inputs is None or len(selected_inputs) < 1:
            origin_vector = np.zeros([len(input_feature_vector_list[0].get_features())])
            for i, input_vector in enumerate(input_feature_vector_list):
                input_feature_values = np.array([value for value in input_vector.get_features().values()])
                distance = calculate_distance(input_feature_values, origin_vector, p_value)
                if max_dist < distance:
                    max_dist = distance
                    max_dist_index = i
            if max_dist_index != -1:
                selected_inputs.add(input_feature_vector_list[max_dist_index])
        else:
            for i, input_vector in enumerate(input_feature_vector_list):
                input_feature_values = np.array([value for value in input_vector.get_features().values()])
                min_dist = float("inf")
                for j, selected_vector in enumerate(selected_inputs):
                    selected_input_feature_values = np.array([value for value in selected_vector.get_features().values()])
                    distance = calculate_distance(input_feature_values, selected_input_feature_values, p_value)
                    if distance < min_dist:
                        min_dist = distance
                if max_dist < min_dist:
                    max_dist = min_dist
                    max_dist_index = i
            if max_dist_index != -1:
                selected_inputs.add(input_feature_vector_list[max_dist_index])
                #print(f"added element: {max_dist_index:02d}, max distance {max_dist}")
            else:
                raise Exception("No new input could be found, though list is not full.")
    
    return selected_inputs

def create_Input_list(input_feature_vector_list, inputs):
    input_subset = set()
    for input_feature_vector in input_feature_vector_list:
        for input in inputs:
            if str(input) == input_feature_vector.test_input:
                input_subset.add(input)
    return input_subset

def collect_feature_values(input_feature_vector_list, feature_name)-> np.array:
    feature_value_array = np.zeros(len(input_feature_vector_list))
    for i, input_feature_vector in enumerate(input_feature_vector_list):
        feature_value_array[i] = input_feature_vector.features[feature_name]
    return feature_value_array

def create_feature_dict(input_vector_list)-> dict():
    features = input_vector_list[0].get_features()
    feature_vectors = dict()
    for feature in features:
        feature_vectors[feature]= collect_feature_values(input_vector_list, feature)
    return feature_vectors

def create_feature_matrix(input_vector_list):
    features = input_vector_list[0].get_features()
    feature_vectors = list()
    for feature in features:
        feature_vectors.append(collect_feature_values(input_vector_list, feature))
    return np.transpose(np.array(feature_vectors)), np.array(features)

if __name__ == "__main__":
    pass