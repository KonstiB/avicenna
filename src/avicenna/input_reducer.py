import numpy as np
import copy
import math

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn import decomposition
from scipy.stats import pearsonr
import pandas as pd
from pandas import DataFrame

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

def determine_input_subset(inputs, grammar, selection_size, use_PCA=False, quantitative_feature_weight=1, metric=1) -> list:
    feature_vector_list = generate_feature_vectors(inputs, grammar, [ExistenceFeature, DerivationFeature, LengthFeature, NumericFeature])
    feature_table = create_feature_table(feature_vector_list, inputs)
    feature_table = remove_NaN_values(feature_table)
    print(f"before dim reduction: {feature_table.shape}")
    feature_table = remove_constant_features(feature_table)
    print(f"removed constant: {feature_table.shape}")
    feature_table = remove_redundant_features(feature_table)
    print(f"removed redundant: {feature_table.shape}")
    feature_table = normalize_features(feature_table, weight=quantitative_feature_weight)

    if use_PCA:
        feature_table = pca(feature_table)
        print(f"after PCA: {feature_table.shape}")

    if isinstance(metric, str):
        if "euclid" in metric.lower():
            metric = 2
        elif "manhatten" in metric.lower() or "city-block" in metric.lower():
            metric = 1
    elif not isinstance(metric, (int, float)):
        raise Exception("given parameter for metric can not be interpreted.")
    selected_input_names = select_dissimilar_inputs(feature_table, selection_size, p_value=metric)

    selected_inputs = create_input_list(selected_input_names, inputs)
    return selected_inputs

def generate_feature_vectors(inputs, grammar, feature_types=[ExistenceFeature, 
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
    for test_input in test_inputs:
            feature_vector = collector.collect_features(test_input)
            #print(f"feature_vector: {feature_vector}")
            feature_list.append(feature_vector)

    return feature_list

def collect_feature_values(feature_vector_list, feature_name)-> np.array:
    feature_value_array = np.zeros(len(feature_vector_list))
    for i, input_feature_vector in enumerate(feature_vector_list):
        feature_value_array[i] = input_feature_vector.features[feature_name]
    return feature_value_array

def create_feature_table(feature_vector_list, inputs) -> DataFrame:
    features = feature_vector_list[0].get_features()
    feature_vectors = list()
    for feature in features:
        feature_vectors.append(collect_feature_values(feature_vector_list, feature))
    
    feature_names = [f"{feature}({feature.type.__name__})" for feature in features]
    input_names = [str(input) for input in inputs]
    data_frame = pd.DataFrame(np.transpose(feature_vectors), columns=feature_names, index=input_names , dtype=float)

    return data_frame

def remove_NaN_values(feature_table: DataFrame):
    return feature_table.replace([np.inf, -np.inf, np.nan, "NaN"], 0.0)


def normalize_features(feature_table: DataFrame, weight: float = 1):
    """normalize column-wise each feature using min-max normalization, which results in feature values ranging between [0:1]"""
    quantitative_rows = feature_table.filter(regex="(int)|(float)", axis="columns")
    
    normalized_feature_table=(quantitative_rows-quantitative_rows.min())/(quantitative_rows.max()-quantitative_rows.min()) * weight

    return normalized_feature_table

def remove_constant_features(feature_table: DataFrame) -> DataFrame:
    """Removes all constant features, meaning features that have the same value for all given inputs, from the 
    input_feature_vector"""
    def single_valued_cols(df):
        df = df.to_numpy() 
        return np.invert((df[0] == df).all(axis=0))
    
    reduced_feature_table = feature_table.loc[:, single_valued_cols(feature_table)]

    return reduced_feature_table

def remove_redundant_features(feature_table: DataFrame) -> DataFrame:
    """removes redundant features, meaning features where every value is the same as for another feature, always keeps the former feature"""    
    # remove redundant features
    feature_table = feature_table.T.drop_duplicates().T
    return feature_table

def pca(feature_table) -> DataFrame:
    feature_matrix = feature_table.to_numpy()
    
    # apply z-score standardization to scale data for pca
    scaler = StandardScaler()
    scaled_feature_matrix = scaler.fit_transform(feature_matrix)

    # perform pca
    pca = decomposition.PCA(n_components='mle', svd_solver='full')
    transformed_feature_matrix = pca.fit_transform(scaled_feature_matrix)
    new_feature_table = pd.DataFrame(transformed_feature_matrix, index=feature_table.index)
    
    return new_feature_table

def calculate_distance(x: np.array, y: np.array, p: float) -> float:
    elem_distance = np.absolute(x - y)
    elem_distance_potentiated = np.power(elem_distance, p)


    summed_elem_distance = np.sum(elem_distance_potentiated)
    distance = np.power(summed_elem_distance, 1/p)
    if math.isnan(distance):
        print(f"IS NAN {x}, {y}")
    return distance

def vector_norms(row_vectors: np.array, p: float) -> float:
    vectors_potentiated = np.power(np.absolute(row_vectors), p)
    return pow(np.sum(vectors_potentiated, axis=1), 1/p)


def select_dissimilar_inputs(feature_table: DataFrame, selection_size: int, preselected_inputs=None, p_value=1):
    if len(feature_table) < selection_size:
        raise Exception("Cannot produce selection that is bigger than base set!")
    
    selected_input_names = preselected_inputs
    if selected_input_names is None:
        selected_input_names = set()
    
    selected_input_table = DataFrame(columns=feature_table.columns)
    for selected_input in selected_input_names:
        selected_input_table.loc[selected_input] = feature_table.loc[selected_input]
        feature_table.drop(selected_input)
    
    for _ in range(selection_size - len(selected_input_names)):
        max_dist = 0
        max_dist_index = -1
        
        if selected_input_names is None or len(selected_input_names) == 0:
            # if no input has been selected so far, choose input with biggest norm
            norms = vector_norms(feature_table.to_numpy(), p_value)
            max_dist_index = np.argmax(norms)
            max_dist = np.max(norms)
        else:
            for i, feature_vector in enumerate(feature_table.to_numpy()):                
                # calculate distances of all inputs to specified feature_vector
                distances = vector_norms(selected_input_table.to_numpy(copy=True) - feature_vector, p_value)
                min_distance = np.min(distances)
                if max_dist < min_distance:
                    max_dist = min_distance
                    max_dist_index = i
        
        if max_dist_index >= 0 and max_dist_index < len(feature_table):
            input = feature_table.index[max_dist_index]
            selected_input_names.add(input)
            selected_input_table.loc[input] = feature_table.loc[input]
            feature_table = feature_table.drop(input)

            #print(f"added element: {input}[{max_dist_index:02d}], max distance {max_dist}, list size: {len(selected_input_names)}/{selected_input_table.shape}, {np.shape(feature_matrix)}/{feature_table.shape}")
        else:
            raise Exception("No new input could be found, though list is not full.")
    
    return selected_input_names

def create_input_list(selected_input_names, inputs):
    input_subset = set()
    for selected_input_name in selected_input_names:
        for input in inputs:
            if str(input) == selected_input_name:
                input_subset.add(input)
    return input_subset

if __name__ == "__main__":
    pass