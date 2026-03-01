import numpy as np
import pandas as pd

def condense_matrix(ss_m, categories: int = 8, instances: int = 4):
    """
    Condenses a representation matrix into a matrix of mean intra-inter category invariance

    Parameters
    ----------
    ss_m : array
        Representation matrix
    categories : int
        Number of categories
    instances : int
        Number of instances per category

    Returns
    -------
    condensed_matrix : array
        Matrix of mean intra-inter category invariance
        shape (categories, categories)
    """

    total_instances = categories * instances
    cat_init = np.arange(0,total_instances,instances)
    cat_end = np.arange(instances,total_instances+1,instances)
    cats = np.arange(0,categories)
    condensed_matrix = np.zeros((categories,categories))
    for slctd_ct in cats:
        row_from_matrix = ss_m[cat_init[slctd_ct]:cat_end[slctd_ct],:]
        mean_per_row = []
        for cat in cats:
            if cat == slctd_ct:
                cat_responses = row_from_matrix[:,cat_init[cat]:cat_end[cat]]
                a,b = np.triu_indices(instances,1)
                mean = []
                for idx in zip(a,b):
                    mean.append(cat_responses[idx[0],idx[1]])
                mean = np.array(mean).mean()
                mean_per_row.append(mean)
            else:
                inter_mean = row_from_matrix[:,cat_init[cat]:cat_end[cat]].mean()
                mean_per_row.append(inter_mean)
        mean_per_row = np.array(mean_per_row)
        condensed_matrix[slctd_ct,:] = mean_per_row
    return condensed_matrix


def get_pair_invariance_df(mtx: np.array):
    cat = ['Leaves', 'Circles', 'Dryland', 'Rocks', 'Tiles', 'Squares', 'Rleaves', 'Paved'] 
    areas = ['V1', 'medial', 'lateral', 'anterior']
    n_features = len(mtx.shape)
    layers = [1, 2] 
    df_pair = pd.DataFrame()
    if n_features == 5:
        for m in range(mtx.shape[0]):
            for i_a, area in enumerate(areas):
                for layer in layers:
                    df = pd.DataFrame()
                    matrix = mtx[m, i_a, layer-1, :, :]
                    condensed = condense_matrix(matrix, 8, 4)
                    a, b = np.triu_indices(8,1)
                    positive_cat = []
                    negative_cat = []
                    invariance_index = []
                    for i in zip(a,b):
                        positive_cat.append(cat[i[0]])
                        negative_cat.append(cat[i[1]])
                        invariance_index.append(np.mean([condensed[i[0],i[0]], condensed[i[1],i[1]]]) - condensed[i[0],i[1]])
                    df["positive_category"] = np.array(positive_cat)
                    df["negative_category"] = np.array(negative_cat)
                    df["pair_invariance"] = np.array(invariance_index)
                    df["area"] = area
                    df["layer"] = layer
                    df["mouse"] = m  
                    df_pair = pd.concat([df_pair, df])
        df_pair.reset_index(inplace=True, drop=True)
    elif n_features == 4:
        for m in range(mtx.shape[0]):
            for i_a, area in enumerate(areas):
                df = pd.DataFrame()
                matrix = mtx[m, i_a, :, :]
                condensed = condense_matrix(matrix, 8, 4)
                a, b = np.triu_indices(8,1)
                positive_cat = []
                negative_cat = []
                invariance_index = []
                for i in zip(a,b):
                    positive_cat.append(cat[i[0]])
                    negative_cat.append(cat[i[1]])
                    invariance_index.append(np.mean([condensed[i[0],i[0]], condensed[i[1],i[1]]]) - condensed[i[0],i[1]])
                df["positive_category"] = np.array(positive_cat)
                df["negative_category"] = np.array(negative_cat)
                df["pair_invariance"] = np.array(invariance_index)
                df["area"] = area
                df["mouse"] = m
                df_pair = pd.concat([df_pair, df])
    else:
        raise ValueError("The input matrix must have 4 or 5 dimensions, got %d dimensions." % n_features)
    return df_pair

def compute_pair_inv_model(rep_mtx, categories:int = 8, instances:int = 4):
    cat = ['Leaves', 'Circles', 'Dryland', 'Rocks', 'Tiles', 'Squares', 'Rleaves', 'Paved'] 
    nlayers = rep_mtx.shape[0]
    alex_df_pair = pd.DataFrame()
    for il in range(nlayers):
        df = pd.DataFrame()
        matrix = rep_mtx[il]
        condensed = condense_matrix(matrix, categories, instances)
        a, b = np.triu_indices(categories,1)
        positive_cat = []
        negative_cat = []
        invariance_index = []
        for i in zip(a,b):
            positive_cat.append(cat[i[0]])
            negative_cat.append(cat[i[1]])
            invariance_index.append(np.mean([condensed[i[0],i[0]], condensed[i[1],i[1]]]) - condensed[i[0],i[1]])
        df["positive_category"] = np.array(positive_cat)
        df["negative_category"] = np.array(negative_cat)
        df["pair_invariance"] = np.array(invariance_index)
        df["layer"] = il
        alex_df_pair = pd.concat([alex_df_pair, df])
    alex_df_pair.reset_index(inplace=True, drop=True)
    return alex_df_pair

def compute_model_rep_mtx(resp):
    firstn_cats = 8
    n_instances = 4
    total_samples = firstn_cats * n_instances
    nlayers = resp.shape[0]
    representation_matrix = np.zeros((nlayers,total_samples, total_samples))
    for il in range(nlayers):
        features = resp[il]
        for i in range(total_samples):
            for j in range(total_samples):
                representation_matrix[il,i,j] = np.corrcoef(features[i], features[j])[0,1]
    return representation_matrix