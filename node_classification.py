import pickle
from load_datasets import get_training_params, load_config
from torch_geometric.datasets import Planetoid, Coauthor,CitationFull,AttributedGraphDataset
import torch_geometric.transforms as T

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical


def tune_on_validation_(X_train, y_train, X_val, y_val, param_grid):
    best_score = -np.inf
    best_model = None

    for C in param_grid['C']:
        for solver in param_grid['solver']:
            model = LogisticRegression(C=C, solver=solver, max_iter=param_grid['max_iter'][0], tol= param_grid['tol'][0] , random_state=42, n_jobs=10)

            model.fit(X_train, y_train)

            val_predictions = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)

            if val_accuracy > best_score:
                best_score = val_accuracy
                best_model = model
        
    best_model = model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0))
    return best_model, best_score


def train_val_test(x_train, x_val, x_test, y_train, y_val, y_test):
    results = []

    accuracies, average_precisions = [], []

    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_val = scaler.transform(x_val)
    X_test = scaler.transform(x_test)

    param_grid = {
        'C': [0.001, 0.1, 10],
        'solver': ['saga'],
        'tol': [1e-3],
        'max_iter': [1000]
    }

    best_model, best_score = tune_on_validation_(X_train, y_train, X_val, y_val, param_grid)
    print(f"Best Validation accuracy: {best_score:.4f}")


    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]


    accuracy = accuracy_score(y_test, y_pred)

    results.append({
        'Accuracy': accuracy,

    })


    results_df = pd.DataFrame(results)
    
    return results_df


if __name__ == "__main__":    

    config = load_config()
    
    path_params = get_training_params(config, 'paths')
    experiment_params = get_training_params(config, 'experiment')
    dataset_name = experiment_params['dataset_name']
 
    print("Loading dataset and embeddings...")
    if dataset_name in ['cora', 'citeseer', 'pubmed']:

        dataset = Planetoid(root="./data", name=dataset_name, transform=T.NormalizeFeatures(), split="public")
    elif dataset_name in ["physics", "cs"]:
        dataset = Coauthor(root="./data", name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name=="dblp":
        dataset = CitationFull(root="./data", name=dataset_name , transform=T.NormalizeFeatures())
    elif dataset_name=="blogcatalog":
        dataset = AttributedGraphDataset(root="./data", name=dataset_name , transform=T.NormalizeFeatures())
    
    for emb in path_params['embeddings']:   
        print("Embedding for file: ", emb)
        with open(emb, 'rb') as pickle_file:
            node_embeddings = pickle.load(pickle_file)              

        if dataset_name in ['cora', 'citeseer', 'pubmed']:
                train_nodes, val_nodes, test_nodes = dataset[0].train_mask, dataset[0].val_mask, dataset[0].test_mask
                train_indices, val_indices, test_indices = train_nodes.nonzero().squeeze(), val_nodes.nonzero().squeeze(), test_nodes.nonzero().squeeze()


                L_train = [i for i in train_indices.tolist() if i in list(node_embeddings.index)]
                L_val = [i for i in val_indices.tolist() if i in list(node_embeddings.index)]
                L_test = [i for i in test_indices.tolist() if i in list(node_embeddings.index)]

        
                Y = pd.DataFrame(dataset[0].y)
                x_train, x_val, x_test = node_embeddings.loc[L_train], node_embeddings.loc[L_val], node_embeddings.loc[L_test]
                y_train, y_val, y_test = Y.loc[L_train], Y.loc[L_val], Y.loc[L_test]



                results_df = train_val_test(x_train, x_val, x_test, y_train, y_val, y_test)
                print()
                print(results_df.to_string())
                print()

        else:

                y = dataset[0].y.numpy()  
                num_classes = len(np.unique(y))

                num_repeats = 10
                results = []
                for _ in range(num_repeats):
                    train_indices_list = []
                    val_indices_list = []
                    test_indices_list = []
                    for c in range(num_classes):
                        class_indices = np.where(y == c)[0]  
                        train_idx, rest_idx = train_test_split(class_indices, train_size=20, stratify=None, random_state=None, shuffle=True)
                        val_idx, test_idx = train_test_split(rest_idx, train_size=30, stratify=None, random_state=None, shuffle=True)
                        
                        train_indices_list.append(train_idx)
                        val_indices_list.append(val_idx)
                        test_indices_list.append(test_idx)
                    
                    train_indices_list = np.concatenate(train_indices_list)
                    val_indices_list = np.concatenate(val_indices_list)
                    test_indices_list = np.concatenate(test_indices_list)


                    Y = pd.DataFrame(dataset[0].y)
                    x_train, x_val, x_test = node_embeddings.loc[train_indices_list], node_embeddings.loc[val_indices_list], node_embeddings.loc[test_indices_list]
                    y_train, y_val, y_test = Y.loc[train_indices_list], Y.loc[val_indices_list], Y.loc[test_indices_list]


                    results_df = train_val_test(x_train, x_val, x_test, y_train[0].values, y_val[0].values, y_test[0].values)
                    results.append(results_df["Accuracy"].values[0])
                print("Results: ", results)
                print("Average accuracy: ", np.mean(results), "+-", np.std(results))
                print("Max accuracy: ", np.max(results))


     