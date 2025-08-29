import pickle
from tqdm import tqdm
from load_datasets import get_training_params, load_config, load_torch_to_networkx, remove_small_components,load_ogbl_to_networkx,load_ogbl_train_val_test
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import networkx as nx
from gensim.models import Word2Vec

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

def compute_mrr(predictions, true_edges):

    reciprocal_ranks = []
    true_edges_set = set(true_edges)  
    for (edge, score) in predictions:
        if edge in true_edges_set:
          
            sorted_predictions = sorted(predictions, key=lambda x: -x[1])
            rank = 1 + [x[0] for x in sorted_predictions].index(edge)
            reciprocal_ranks.append(1.0 / rank)
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def compute_mpr(predictions, true_edges, k):
  
    precision_scores = []
    true_edges_set = set(true_edges)  
    for (edge, score) in predictions:
        if edge in true_edges_set:

            top_k = [x[0] for x in sorted(predictions, key=lambda x: -x[1])[:k]]
            correct_in_top_k = len(set(top_k).intersection(true_edges_set))
            precision_scores.append(correct_in_top_k / k)
    return np.mean(precision_scores) if precision_scores else 0.0

def compute_precision_at_k(predictions, true_edges, k=1):
   
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    top_k_predictions = [edge for edge, _ in sorted_predictions[:k]]

    true_positives = len(set(top_k_predictions).intersection(true_edges))
    precision_at_k = true_positives / k
    return precision_at_k


def compute_hits_at_k(predictions, true_edges, k=1):
    predictions_df = pd.DataFrame(predictions, columns=['edge', 'score'])
    predictions_df[['source', 'target']] = pd.DataFrame(predictions_df['edge'].tolist(), index=predictions_df.index)

    grouped = (
        predictions_df.groupby('source')  
        [['score', 'target']]  
        .apply(lambda x: x.nlargest(k, 'score'))  
    )


    top_k_predictions = grouped.groupby('source')['target'].apply(list).to_dict()

    true_edges_df = pd.DataFrame(list(true_edges), columns=['source', 'target'])

    true_edges_df['hit'] = true_edges_df.apply(
        lambda row: row['target'] in top_k_predictions.get(row['source'], []), axis=1
    )

    hits_at_k = true_edges_df['hit'].mean()
    return hits_at_k

def compute_attention_weights(embedding1, embedding2):

    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    attention_score = np.dot(embedding1, embedding2)
    
    attention_scores = np.array([attention_score, 1 - attention_score]) 
    attention_weights = softmax(attention_scores)
    
    return attention_weights

def softmax(x):

    e_x = np.exp(x - np.max(x))  
    return e_x / e_x.sum()



def generate_edge_embeddings(G, node_embeddings, method='hadamard'):

    edge_data = []
  
    for edge in G.edges():
        node1, node2 = edge
        edge_embedding = get_edge_embedding(node1, node2, node_embeddings, method=method)
        edge_data.append({
            'node1': node1,
            'node2': node2,
            'edge_embedding': edge_embedding
        })
    
    return edge_data


def generate_negative_edges(graph, num_negative_edges):
    existing_edges = set(graph.edges())
    negative_edges = set()
    
    while len(negative_edges) < num_negative_edges:
        node1, node2 = tuple(np.random.choice(graph.nodes(), size=2, replace=False))
        if (node1, node2) not in existing_edges and (node2, node1) not in existing_edges:
            if (node1, node2) not in negative_edges and (node2, node1) not in negative_edges:
                negative_edges.add((node1, node2))
    
    return list(negative_edges)

def get_edge_embedding(node1, node2, node_embeddings, method='hadamard'):
    edge_embedding =  None
    embedding1 = node_embeddings.loc[node1].values
    embedding2 = node_embeddings.loc[node2].values

    if method == 'hadamard':
        edge_embedding = embedding1 * embedding2
    elif method == 'average':
        edge_embedding = (embedding1 + embedding2) / 2
    elif method == 'cosine':
        edge_embedding = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    elif method == 'max':
        edge_embedding = np.maximum(embedding1, embedding2)
    elif method == 'min':
        edge_embedding = np.minimum(embedding1, embedding2)
    elif method == 'concatenation':
        edge_embedding = np.concatenate((embedding1, embedding2), axis=0)
    elif method == 'L1':
        edge_embedding = np.abs(embedding1 - embedding2)
    elif method == 'L2':
        edge_embedding = (embedding1 - embedding2) ** 2
    elif method == 'attention':
        attention_weights = compute_attention_weights(embedding1, embedding2)
        edge_embedding = attention_weights[0] * embedding1 + attention_weights[1] * embedding2
    
    return edge_embedding

def generate_negative_edge_embeddings(negative_edges, node_embeddings, method='hadamard'):

    negative_edge_data = []
   
    for edge in negative_edges:
        node1, node2 = edge
        edge_embedding = get_edge_embedding(node1, node2, node_embeddings, method=method)
        negative_edge_data.append({
            'node1': node1,
            'node2': node2,
            'edge_embedding': edge_embedding
        })
    
    return negative_edge_data

def evaluate_classifiers_with_cv(X, edges, y, classifiers, n_splits=10, random_state=42):

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    for classifier_name, classifier in classifiers.items():
        classifier = eval(classifier)
        print(f"Evaluating {classifier_name}...")

        accuracies, aucs, precisions, avg_precisions, hits_at_1, hits_at_3, hits_at_10 = [], [], [], [], [], [], []
        mrrs, mprs = [], []  
        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            edges_train, edges_test = edges[train_index], edges[test_index]
            y_train, y_test = y[train_index], y[test_index]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)
            y_pred_prob = classifier.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_prob)
            precision = precision_score(y_test, y_pred)
            avg_precision = average_precision_score(y_test, y_pred_prob)

            accuracies.append(accuracy)
            aucs.append(auc)
            precisions.append(precision)
            avg_precisions.append(avg_precision)

            predictions = [(tuple(edge), score) for edge, score in zip(edges_test, y_pred_prob)]

            true_edges = set(tuple(edge) for edge, label in zip(edges_test, y_test) if label == 1)

            hits_at_1.append(compute_hits_at_k(predictions, true_edges, k=1))
            hits_at_3.append(compute_hits_at_k(predictions, true_edges, k=3))
            hits_at_10.append(compute_hits_at_k(predictions, true_edges, k=10))
            

            mrrs.append(compute_mrr(predictions, true_edges))
            mprs.append(compute_mpr(predictions, true_edges, k=10))  


        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        avg_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        avg_precision = np.mean(precisions)
        std_precision = np.std(precisions)

        avg_avg_precision = np.mean(avg_precisions)
        std_avg_precision = np.std(avg_precisions)
        
        avg_hits_at_1 = np.mean(hits_at_1)
        std_hits_at_1 = np.std(hits_at_1)

        avg_hits_at_3 = np.mean(hits_at_3)
        std_hits_at_3 = np.std(hits_at_3)

        avg_hits_at_10 = np.mean(hits_at_10)
        std_hits_at_10 = np.std(hits_at_10)

        avg_mrr = np.mean(mrrs)
        std_mrr = np.std(mrrs)

        avg_mpr = np.mean(mprs)
        std_mpr = np.std(mprs)

        results.append({
            'Classifier': classifier_name,
            'Accuracy': f"{avg_accuracy:.4f} ± {std_accuracy:.4f}",
            'AUC': f"{avg_auc:.4f} ± {std_auc:.4f}",
            'Precision': f"{avg_precision:.4f} ± {std_precision:.4f}",
            'Average Precision': f"{avg_avg_precision:.4f} ± {std_avg_precision:.4f}",
            'HITS@1': f"{avg_hits_at_1:.4f} ± {std_hits_at_1:.4f}",
            'HITS@3': f"{avg_hits_at_3:.4f} ± {std_hits_at_3:.4f}",
            'HITS@10': f"{avg_hits_at_10:.4f} ± {std_hits_at_10:.4f}",
            'MRR': f"{avg_mrr:.4f} ± {std_mrr:.4f}",
            'MPR@10': f"{avg_mpr:.4f} ± {std_mpr:.4f}",
        })

    results_df = pd.DataFrame(results)
    
    return results_df

def tune_on_validation_(X_train, y_train, X_val, y_val, param_grid):
    best_score = -np.inf
    best_model = None

    for C in param_grid['C']:
        for solver in param_grid['solver']:
            model = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42, n_jobs=10)
            
            model.fit(X_train, y_train)
            
            val_predictions = model.predict(X_val)
            val_auc = roc_auc_score(y_val, val_predictions)
            
            if val_auc > best_score:
                best_score = val_auc
                best_model = model
    
    return best_model, best_score



def tune_on_validation(X_train, y_train, X_val, y_val, param_grid):

    
    search = HalvingGridSearchCV(
        LogisticRegression(n_jobs=-1),
        param_grid,
        factor=3,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,  
    )
    
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_estimator_.score(X_val, y_val)

def train_val_test(X, edges, Y, classifiers, sizes=(0.7,0.1,0.2), repeat = 10):

    train_ratio = sizes[0]
    validation_ratio = sizes[1]
    test_ratio = sizes[2]



    results = []

    for classifier_name, classifier in classifiers.items():
        print(f"Evaluating {classifier_name}...")

        accuracies, aucs, precisions, avg_precisions, hits_at_1, hits_at_3, hits_at_10 = [], [], [], [], [], [], []
        mrrs, mprs = [], []  
        print(X.shape)
        for i in tqdm(range(repeat), desc="train, validation, test iterations", total=repeat):

            x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=train_ratio, shuffle=True, stratify=Y)

            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=True, stratify=y_test) 
            
            
            edges_train, edges_val, edges_test = edges[y_train.index.values], edges[y_val.index.values], edges[y_test.index.values]


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

            best_model, best_val_auc = tune_on_validation(X_train, y_train, X_val, y_val, param_grid)




            y_pred = best_model.predict(X_test)
            y_pred_prob = best_model.predict_proba(X_test)[:, 1]


            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_prob)
            precision = precision_score(y_test, y_pred)
            avg_precision = average_precision_score(y_test, y_pred_prob)


            accuracies.append(accuracy)
            aucs.append(auc)
            precisions.append(precision)
            avg_precisions.append(avg_precision)


            predictions = [(tuple(edge), score) for edge, score in zip(edges_test, y_pred_prob)]
  
            true_edges = set(tuple(edge) for edge, label in zip(edges_test, y_test) if label == 1)

            hits_at_1.append(compute_hits_at_k(predictions, true_edges, k=1))
            hits_at_3.append(compute_hits_at_k(predictions, true_edges, k=3))
            hits_at_10.append(compute_hits_at_k(predictions, true_edges, k=10))
            

            mrrs.append(compute_mrr(predictions, true_edges))
            mprs.append(compute_mpr(predictions, true_edges, k=10))  


        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        avg_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        avg_precision = np.mean(precisions)
        std_precision = np.std(precisions)

        avg_avg_precision = np.mean(avg_precisions)
        std_avg_precision = np.std(avg_precisions)
        
        avg_hits_at_1 = np.mean(hits_at_1)
        std_hits_at_1 = np.std(hits_at_1)

        avg_hits_at_3 = np.mean(hits_at_3)
        std_hits_at_3 = np.std(hits_at_3)

        avg_hits_at_10 = np.mean(hits_at_10)
        std_hits_at_10 = np.std(hits_at_10)

        avg_mrr = np.mean(mrrs)
        std_mrr = np.std(mrrs)

        avg_mpr = np.mean(mprs)
        std_mpr = np.std(mprs)


        results.append({
            'Classifier': classifier_name,
            'Accuracy': f"{avg_accuracy:.4f} ± {std_accuracy:.4f}",
            'AUC': f"{avg_auc:.4f} ± {std_auc:.4f}",
            'Precision': f"{avg_precision:.4f} ± {std_precision:.4f}",
            'Average Precision': f"{avg_avg_precision:.4f} ± {std_avg_precision:.4f}",
            'HITS@1': f"{avg_hits_at_1:.4f} ± {std_hits_at_1:.4f}",
            'HITS@3': f"{avg_hits_at_3:.4f} ± {std_hits_at_3:.4f}",
            'HITS@10': f"{avg_hits_at_10:.4f} ± {std_hits_at_10:.4f}",
            'MRR': f"{avg_mrr:.4f} ± {std_mrr:.4f}",
            'MPR@10': f"{avg_mpr:.4f} ± {std_mpr:.4f}",
        })


    results_df = pd.DataFrame(results)
    
    return results_df

def train_val_test_ogbl(X, edges, Y, classifiers):
    load_ogbl_train_val_test


    return results_df


if __name__ == "__main__":    

   
    config = load_config()
    
    path_params = get_training_params(config, 'paths')
    experiment_params = get_training_params(config, 'experiment')
    dataset_name = experiment_params['dataset_name']
    training_params = get_training_params(config, 'training_params')

    print("Loading dataset and embeddings...")
    if dataset_name.startswith('ogbl'):
        G = load_ogbl_to_networkx(root=path_params['data'], name=dataset_name)
    else:
        G = load_torch_to_networkx(root=path_params['data'], name=dataset_name)

    

    num_negative_edges = len(G.edges())
    print(f"Generating {num_negative_edges} negative edges...")
    negative_edges = generate_negative_edges(G, num_negative_edges)
    
    file_exists = False
    for emb_metric in path_params['embeddings']:
        print("Embedding for file: ", emb_metric)

        if not os.path.exists(emb_metric):
            raise FileNotFoundError(f"Embedding file not found: {emb_metric}")
        with open(emb_metric, 'rb') as pickle_file:
            node_embeddings = pickle.load(pickle_file)                
            if isinstance(node_embeddings, Word2Vec):
                node_embeddings = pd.DataFrame(np.array([node_embeddings.wv[node] for node in node_embeddings.wv.index_to_key]), index=node_embeddings.wv.index_to_key)

        for method in experiment_params['edge_aggregation_methods']:
            print(f"Edge embedding method: {method}")

            edge_data = generate_edge_embeddings(G, node_embeddings, method=method)
            edge_df = pd.DataFrame(edge_data)

            negative_edge_data = generate_negative_edge_embeddings(negative_edges, node_embeddings, method=method)
            negative_edge_df = pd.DataFrame(negative_edge_data)

            edge_df['label'] = 1
            negative_edge_df['label'] = 0

            full_edge_df = pd.concat([edge_df, negative_edge_df])

            if full_edge_df['edge_embedding'].apply(lambda x: np.isnan(x).any()).any():
                print("NaN values found in edge embeddings. Handling them...")
                
            X = np.array(list(full_edge_df['edge_embedding']))
            X = np.vstack(X) 
            y = full_edge_df['label']

            edges = full_edge_df[['node1', 'node2']].apply(tuple, axis=1).values


            if dataset_name.startswith('ogbl'):
                results_df = train_val_test_ogbl(X, edges, y, experiment_params['classifiers'])
            else:
                results_df = train_val_test(X, edges, y, experiment_params['classifiers'], sizes=(0.85,0.05,0.1), repeat = 10)


            if not file_exists:

                results_df.to_csv('./data/results_prediction/'+dataset_name+'results.tsv', sep='\t', index=False)
                file_exists = True
            else:

                results_df.to_csv('./data/results_prediction/'+dataset_name+'results.tsv', mode='a', sep='\t',header=False, index=False)
            print()
            print(results_df.to_string())
        print()