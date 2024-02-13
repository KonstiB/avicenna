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

def determine_input_subset(inputs, 
                           grammar, 
                           selection_size, 
                           mode="dissimilar", 
                           feature_types=[ExistenceFeature, DerivationFeature, LengthFeature, NumericFeature], 
                           use_PCA=False, 
                           quantitative_feature_weight=1, 
                           metric=1) -> list:
    

    feature_vector_list = generate_feature_vectors(inputs, grammar, feature_types)
    feature_table = create_feature_table(feature_vector_list, inputs)
    feature_table = remove_NaN_values(feature_table)
    #print(f"before dim reduction: {feature_table.shape}")
    feature_table = remove_constant_features(feature_table)
    #print(f"removed constant: {feature_table.shape}")
    feature_table = remove_redundant_features(feature_table)
    #print(f"removed redundant: {feature_table.shape}")
    feature_table = normalize_features(feature_table, weight=quantitative_feature_weight)
    #print(f"after normalization: {feature_table.shape}")

    if use_PCA:
        feature_table = pca(feature_table)
        #print(f"after PCA: {feature_table.shape}")

    if isinstance(metric, str):
        if "euclid" in metric.lower():
            metric = 2
        elif "manhatten" in metric.lower() or "city-block" in metric.lower():
            metric = 1
    elif not isinstance(metric, (int, float)):
        raise Exception("given parameter for metric can not be interpreted.")
    selected_input_names = select_input_subset(feature_table, selection_size, p_value=metric, mode= mode)

    selected_inputs = create_input_list(selected_input_names, inputs)
    
    #print(f"at the end: {feature_table.shape}")
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
    qualitative_rows = feature_table.drop(columns=quantitative_rows, errors="ignore")
    
    normalized_feature_table=(quantitative_rows-quantitative_rows.min())/(quantitative_rows.max()-quantitative_rows.min()) * weight

    return pd.concat([normalized_feature_table, qualitative_rows], axis=1)

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

def select_input_subset(feature_table: DataFrame, selection_size: int, preselected_inputs=None, p_value=1, mode = "dissimilar"):
    
    def norm_vectors(row_vectors: np.array, p: float):
        if isinstance(row_vectors, pd.DataFrame):
            row_vectors = row_vectors.to_numpy()
        if float(p) == 1.0:
            return np.sum(np.absolute(row_vectors), axis=1)
        else:  
            vectors_potentiated = np.power(np.absolute(row_vectors), p)
            return np.power(np.sum(vectors_potentiated, axis=1), 1.0/float(p))

    def pairwise_distances(vectors, p_value):
        if isinstance(vectors, pd.DataFrame):
            vectors = vectors.to_numpy()
        num_vectors = len(vectors)
        distances = np.zeros((num_vectors, num_vectors), dtype=float)
        for i in range(num_vectors):
                distances[i] = norm_vectors(vectors[i] - vectors, p_value)
        
        np.fill_diagonal(distances, float("nan"))
        return distances

    
    if len(feature_table) < selection_size:
        raise Exception("Cannot produce selection that is bigger than base set!")
    
    selected_input_names = preselected_inputs
    if selected_input_names is None:
        selected_input_names = set()
    
    selected_input_index = [feature_table.index.get_loc(name) for name in selected_input_names]
    
    # preprocess all distances and norms to save time
    vector_norms = norm_vectors(feature_table, p_value)
    vector_pair_distances = pairwise_distances(feature_table.to_numpy(), p_value=p_value)

    for selection_num in range(selection_size - len(selected_input_index)):
        best_dist = 0
        best_dist_index = -1
        
        if selected_input_index is None or len(selected_input_index) == 0:
            # if no input has been selected so far, choose input with biggest/smallest norm
            if mode == "dissimilar":
                best_dist_index = np.argmax(vector_norms)
                best_dist = np.max(vector_norms)
            elif mode == "similar":
                best_dist_index = np.argmin(vector_norms)
                best_dist = np.min(vector_norms)
        else:
            #[x for x in main_list if x not in elements_to_remove]
            for i in [index for index in range(len(feature_table.index)) if index not in selected_input_index]:                
                # retrieve distances of all selected inputs to specified feature_vector
                vector_index = [i] * len(selected_input_index)
                distances = vector_pair_distances[vector_index, selected_input_index]
                if mode == "dissimilar": 
                    min_distance = np.nanmin(distances)
                    if best_dist < min_distance:
                        best_dist = min_distance
                        best_dist_index = i
                elif mode == "similar":
                    max_distance = np.nanmax(distances)
                    if best_dist > max_distance:
                        best_dist = max_distance
                        best_dist_index = i

        if best_dist_index >= 0 and best_dist_index < len(feature_table):
            selected_input_index.append(best_dist_index)
            input = feature_table.index[best_dist_index]
            selected_input_names.add(input)
            #selected_input_table.loc[input] = feature_table.loc[input]
            #feature_table = feature_table.drop(input)

        else:
            raise Exception(f"""No new input could be found, though list is not full.\n Index:{best_dist_index}, best distance: {best_dist}\nCurrent iteration: {selection_num}, Max iterations: {selection_size}""")
    
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