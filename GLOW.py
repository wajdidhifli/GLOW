import networkx as nx
import numpy as np
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from load_datasets import get_training_params, load_config, load_torch_to_networkx, load_ogbl_to_networkx, remove_small_components
import random, pickle, math
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

import pandas as pd


class HybridGraphEmbedding(nn.Module):
    def __init__(self, graph, num_walks=10,  walk_length=10, num_landmarks=10, num_candidate_landmarks=10,  embedding_dim=128, learning_rate=0.01, num_epochs=50, patience=25, alpha=0, workers = 4):
        super(HybridGraphEmbedding, self).__init__() 
        self.graph = graph
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.final_embeddings = None
        self.loss_history = []  
        self.best_loss = float('inf') 
        self.epochs_without_improvement = 0  
        self.patience = patience  
        if workers >= cpu_count():
            self.workers = cpu_count() - 1
        else:
            self.workers = workers
        self.penalty = walk_length
        self.used_nodes = set()  
        self.num_landmarks = num_landmarks
        self.num_candidate_landmarks = num_candidate_landmarks
        self.nodes = list(self.graph.nodes()) 
        self.alpha = alpha

    def generate_walks(self):

        if self.walk_length <= 0:
            raise ValueError("walk_length must be a positive integer")

        nodes = list(self.graph.nodes())
        nb_workers = self.workers
        if nb_workers == -1:
            nb_workers = cpu_count()
        node_sets = np.array_split(nodes, nb_workers)

        print(f"Generating random walks from each node ({len(nodes)} nodes)...")
        walks = Parallel(n_jobs=self.workers)(
            delayed(self.process_node_set)(node_set, f"Generating walks (CPU: {i+1})") for i, node_set in enumerate(node_sets)
        )

        walks = [walk for sublist in walks for walk in sublist]

        additional_walks = self.num_walks - len(nodes)
        if additional_walks > 0:
            print("Generating additional random walks...")
            additional_walks_list = Parallel(n_jobs=self.workers)(
                delayed(self.random_walk)(random.choice(nodes)) for _ in tqdm(range(additional_walks), desc="Additional Walks")
            )
            walks.extend(additional_walks_list)

        return walks

    def process_node_set(self, node_set, worker_name):

        walks = []
        for node in tqdm(node_set, desc=worker_name):
            walks.append(self.random_walk(node))
        return walks

    def random_walk(self, start_node):

        walk = [start_node]
        current_node = start_node
        for _ in range(self.walk_length - 1):
            neighbors = list(self.graph.neighbors(current_node))
            if not neighbors:  
                break
            current_node = random.choice(neighbors)
            walk.append(current_node)
        return walk

    def add_fictive_node(self):

        modified_graph = self.graph.copy()
        

        fictive_node = "F"
        modified_graph.add_node(fictive_node)

        components = list(nx.connected_components(self.graph))
        for component in components:

            highest_degree_node = max(component, key=lambda x: self.graph.degree(x))
            modified_graph.add_edge(fictive_node, highest_degree_node)

        return modified_graph


    def compute_jaccard_distance(self, node_walk_sets, node1, node2):

        set1, set2 = node_walk_sets[node1], node_walk_sets[node2]
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccard_similarity = intersection / union if union > 0 else 0
        return 1 - jaccard_similarity
    
    def compute_landmark_similarity_parallel_(self, landmarks):


        modified_graph = self.add_fictive_node()

        nb_workers = self.workers
        if nb_workers == -1:
            nb_workers = cpu_count()


        dists_landmarks = Parallel(n_jobs=self.workers)(
            delayed(self.process_sim_landmark_to_nodes)(modified_graph, landmark, f"Computing distances to landmark : {i+1}") for i, landmark in enumerate(landmarks)
        )

        dists_landmarks = torch.stack(dists_landmarks).T

        del modified_graph

        return dists_landmarks
    
    def process_sim_landmark_to_nodes(self, modified_graph, landmark, landmark_name):
        print(landmark_name)

        paths = nx.single_target_shortest_path(modified_graph, landmark, cutoff=None)
        
        dists = torch.zeros(len(self.nodes))
        for node in self.nodes:
            if "F" in paths[node]:
                dists[node] = len(paths[node]) + self.penalty - 1
            else:
                dists[node] = len(paths[node]) - 1
        return dists


    def compute_landmark_similarity_parallel(self, landmarks):
        """Compute similarity of all nodes to a set of landmark nodes."""

        modified_graph = self.add_fictive_node()

        nodes = list(self.graph.nodes())
        nb_workers = self.workers
        if nb_workers == -1:
            nb_workers = cpu_count()
        node_sets = np.array_split(nodes, nb_workers)

        dists_landmarks = Parallel(n_jobs=self.workers)(
            delayed(self.process_sim_landmarks_node_set)(node_set, landmarks, modified_graph, f"Computing nodes to landmarks distances (CPU: {i+1})") for i, node_set in enumerate(node_sets)
        )


        dists_landmarks = [row for tensor in dists_landmarks for row in tensor]
        dists_landmarks = torch.stack(dists_landmarks)


        if torch.all(dists_landmarks == 0):  
            raise ValueError("All distances with landmarks are 0. This could be because all landmarks are disconnected making all shortest_path_length=0.")
        if torch.isnan(dists_landmarks).any():  
            raise ValueError("All distances with landmarks not NaNs.")

        del modified_graph

        return dists_landmarks

    def process_sim_landmarks_node_set(self, node_set, landmarks, modified_graph, worker_name):

        dists_landmarks_node_set = torch.zeros(len(node_set), len(landmarks))
        for i, node in enumerate(tqdm(node_set, desc=worker_name)):
            dists = torch.zeros(len(landmarks))
            for j, landmark in enumerate(landmarks):
                path = nx.shortest_path(modified_graph, source=node, target=landmark)
                dist = len(path) - 1  
                if "F" in path:
                    dist = self.penalty + len(path)  
                dists[j] = dist
            dists_landmarks_node_set[i] = dists
        return dists_landmarks_node_set
    
    def process_sim_landmarks_node_set__(self, node_set, landmarks, modified_graph, worker_name):

        dists_landmarks_node_set = torch.zeros(len(node_set), len(landmarks))
        for i, node in enumerate(tqdm(node_set, desc=worker_name)):
            dists_landmarks_node_set[i] = self.compute_dist_landmarks_node(node.item(), landmarks, modified_graph)
        return dists_landmarks_node_set

    def compute_dist_landmarks_node(self, node, landmarks, modified_graph):
        dists = torch.zeros(len(landmarks))
        for j, landmark in enumerate(landmarks):
            try:
                path = nx.shortest_path(modified_graph, source=node, target=landmark)
                dist = len(path) - 1  
                if "F" in path:
                    dist = self.penalty + len(path)  
            except nx.NetworkXNoPath:
                dist = 0  
            dists[j] = dist
        return dists 

    def select_farthest_node(self, landmarks, component):

        available_nodes = list(component - self.used_nodes)
        
        if len(available_nodes) < self.num_candidate_landmarks:
            print("Resetting used_nodes set: Not enough unused nodes left.")
            self.used_nodes = set()  
            available_nodes = list(component)  
        
        
        random_nodes = random.sample(available_nodes, min(self.num_candidate_landmarks, len(available_nodes)))
        
        
        results = Parallel(n_jobs=self.workers)(
            delayed(self.compute_min_distance)(node, landmarks) for node in random_nodes
        )
        
        
        distances = dict(results)
        
        
        farthest_node = max(random_nodes, key=lambda x: distances.get(x, float('inf')))
        
        return farthest_node
    
       
    def compute_min_distance(self, node, landmarks):
        min_distance = float('inf')
        for landmark in landmarks:
            try:
                distance = nx.shortest_path_length(self.graph, source=landmark, target=node)
                if distance < min_distance:
                    min_distance = distance
            except nx.NetworkXNoPath:

                continue
        return node, min_distance
    
    def select_landmarks_in_component(self, component, component_num_landmarks):

        landmarks = []
        a = self.graph.subgraph(component)
        selected_landmark = max(component, key=lambda x: a.degree(x))

        landmarks.append(selected_landmark)

        for i in tqdm(range(0, component_num_landmarks), desc="Selecting landmarks", unit="landmark"):
            if i==0:
                continue 
            selected_landmark = self.select_farthest_node([selected_landmark], component)
            landmarks.append(selected_landmark)
            self.used_nodes.add(selected_landmark)
        return landmarks
    
    def select_landmarks(self):

        components = list(nx.connected_components(self.graph))
        print("Number of connected components:", len(components))

        components = [comp for comp in components]
        num_components = len(components)

        if self.num_landmarks < num_components:
            print(f"Warning: self.num_landmarks ({self.num_landmarks}) is less than the number of components ({num_components}). "
                f"Setting self.num_landmarks to {num_components}.")
            self.num_landmarks = num_components
        
        
        component_sizes = torch.tensor([len(comp) for comp in components], dtype=torch.float32)
        total_nodes = component_sizes.sum()
        
        proportions = component_sizes / total_nodes
        components_num_landmarks = torch.floor(self.num_landmarks * proportions).to(torch.int32)
        
        components_num_landmarks += 1
        
        remaining = self.num_landmarks - components_num_landmarks.sum().item()
        if remaining > 0:
            largest_indices = torch.argsort(component_sizes, descending=True)[:remaining]
            components_num_landmarks[largest_indices] += 1
        
        print("Component sizes and landmark allocation:")
        for i, comp in enumerate(components):
            print(f"Component {i + 1}: Size = {len(comp)}, Landmarks = {components_num_landmarks[i].item()}")

        landmarks = Parallel(n_jobs=self.workers)(
            delayed(self.select_landmarks_in_component)(comp, components_num_landmarks[i].item())
            for i, comp in tqdm(enumerate(components), desc="Selecting landmarks", unit="component")
        )
        
        landmarks = [node for sublist in landmarks for node in sublist]
        return landmarks

    def select_landmarks_(self):

            components = list(nx.connected_components(self.graph))
            
            components = [comp for comp in components]
            num_components = len(components)

            if self.num_landmarks < num_components:
                print(f"Warning: self.num_landmarks ({self.num_landmarks}) is less than the number of components ({num_components}). "
                    f"Setting self.num_landmarks to {num_components}.")
                self.num_landmarks = num_components
            
            component_sizes = np.array([len(comp) for comp in components])
            
            proportions = component_sizes / len(self.nodes)
            components_num_landmarks = np.ceil(self.num_landmarks * proportions)
            
            print("Component sizes and landmark allocation:")
            for i, comp in enumerate(components):
                print(f"Component {i + 1}: Size = {len(comp)}, Landmarks = {components_num_landmarks[i]}")
            
            
            landmarks = []
            for i, component in enumerate(components):        

                a = self.graph.subgraph(component)
                selected_landmark = max(component, key=lambda x: a.degree(x))

                landmarks.append(selected_landmark)
                for i in tqdm(range(0, int(components_num_landmarks[i])), desc="Selecting landmarks", unit="landmark"):
                    if i==0:
                        continue 
                    selected_landmark = self.select_farthest_node([selected_landmark], component)
                    landmarks.append(selected_landmark)
                    self.used_nodes.add(selected_landmark)


            return landmarks

    def compute_similarity_embeddings(self, embeddings, landmarks):

        landmark_indices = torch.tensor(
            [self.nodes.index(landmark) for landmark in landmarks],
            device=embeddings.device
        )
        

        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        
        landmark_embeddings = embeddings_normalized[landmark_indices]  
        sim_matrix = torch.mm(embeddings_normalized, landmark_embeddings.T)  
        
        sim_matrix = torch.sub(1, sim_matrix)
        sim_matrix = torch.clamp(sim_matrix, min=1e-8)
        
        return sim_matrix
    

    def min_max_normalize(self, tensor):

        tensor_min = tensor.min()
        tensor_max = tensor.max()
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        return normalized_tensor
    
    def  normalise_to_1(self, p):
        return p/p.sum() 
       
    def kl_divergence(self, p, q):
        assert torch.all(p>=0) and torch.all(q>=0), "Probabilities must be non-negative"
        

        epsilon = 1e-10
        p = p + epsilon 
        q = q + epsilon
        p = p/p.sum()
        q = q/q.sum()
        kl = F.kl_div(torch.log(q), p, reduction='sum')

        if self.alpha > 0:
            Y = torch.tensor([self.graph.nodes[node]["label"] for node in self.nodes]).to(device)
            unique_classes = torch.unique(Y).to(device)

            kl_class_loss = 0.0
            
            for cls in unique_classes:
                indexes = (Y == cls).nonzero(as_tuple=True)[0]
               
                mu = q[indexes].mean()
                sigma = q[indexes].std()
                kl_class_loss += 0.5 * (mu ** 2 + sigma ** 2 - 2 * torch.log(sigma) - 1)

            kl = ((1-self.alpha)*kl) + (self.alpha * kl_class_loss)


        return kl

    def print_parameter_names(self):

        print("Parameter Names:")
        for name, param in self.named_parameters():
            print(f"name={name}, param= {param}")
    
    def fit(self, init_embedding = None, walks=None):

        if walks is None:
            random_walks = self.generate_walks()
        else:
            random_walks = walks


        if init_embedding is None:
            print("Training Word2Vec embeddings...")
            w2v_model = Word2Vec(random_walks, min_count=1, vector_size=self.embedding_dim, window=10, sg=1, negative=5, workers=self.workers, epochs=10)
        else:
            w2v_model = init_embedding
        initial_embeddings = torch.tensor(np.array([w2v_model.wv[node] for node in w2v_model.wv.index_to_key]), dtype=torch.float32).to(device)


        landmarks = self.select_landmarks()
        
        with torch.no_grad():  
            sim_matrix_global  = self.normalise_to_1(self.min_max_normalize(self.compute_landmark_similarity_parallel(landmarks))).to(device)

        embeddings = nn.Parameter(initial_embeddings, requires_grad=True) 
        optimizer = optim.AdamW([embeddings], lr=self.learning_rate, betas=(0.9, 0.999))
        embeddings.to(device)


        total_loss_lst = list()

        use_amp = device.type == "cuda"
        if use_amp:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  
          
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
                
            sim_matrix_embedding_global = self.normalise_to_1(self.compute_similarity_embeddings(embeddings, landmarks)).to(device)
            

            if use_amp:
                with torch.cuda.amp.autocast(enabled=use_amp): 
                    loss = self.kl_divergence(sim_matrix_global, sim_matrix_embedding_global).to(device)
            else:
                loss = self.kl_divergence(sim_matrix_global, sim_matrix_embedding_global).to(device)

            loss_v = loss.item()
            total_loss_lst.append(loss_v)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            
            if loss.item() < self.best_loss:   
            
                self.best_loss = loss.item()
                self.epochs_without_improvement = 0  
            else:
                self.epochs_without_improvement += 1

            if epoch % 50 == 0:

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Total Loss: {loss.item()}")


            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in loss at {loss.item()}.")
                break
            
        self.final_embeddings = pd.DataFrame(embeddings.detach().cpu().numpy(), index=self.nodes)

        return self.final_embeddings, total_loss_lst

    def get_embeddings(self):
      
        if self.final_embeddings is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.final_embeddings



def is_load(embedding_model,  dataset_name, load = False):    
    if not load:
        walks = embedding_model.generate_walks()
        with open('./data/'+dataset_name+'/walks.pkl', 'wb') as file:
            pickle.dump(walks, file)
        print("Training Word2Vec embeddings...")
        init_embeddings = Word2Vec(walks, min_count=1, vector_size=embedding_model.embedding_dim, workers=embedding_model.workers, epochs=1)

        with open('./data/'+dataset_name+'/w2v_embeddings.pkl', 'wb') as file:
            pickle.dump(init_embeddings, file)

    else:
        with open('./data/'+dataset_name+'/w2v_embeddings.pkl', 'rb') as file:
            init_embeddings = pickle.load(file)
        with open('./data/'+dataset_name+'/walks.pkl', 'rb') as file:
            walks = pickle.load(file)

    return init_embeddings, walks

def by_landmark(embedding_model, init_embeddings, walks, dataset_name):
    print("Training with landmark-based global similarity...")
    final_embeddings_landmark, loss = embedding_model.fit(init_embedding=init_embeddings, walks=walks)

    print("Final Embeddings Shape (Landmarks):", final_embeddings_landmark.shape)
    with open('./data/'+dataset_name+'/'+dataset_name+'_landmark_embeddings_'+str(embedding_model.num_landmarks)+'.pkl', "wb") as f:
        pickle.dump(final_embeddings_landmark, f)


def main():

    config = load_config()
    experiment_params = get_training_params(config, 'experiment')
    path_params = get_training_params(config, 'paths')
    dataset_name = experiment_params["dataset_name"]
    root = path_params["data"]

    print(dataset_name)
    if dataset_name.startswith("ogbl-"):
        graph = load_ogbl_to_networkx(root=root, name=dataset_name)
        dataset_name = dataset_name.replace("ogbl-", "ogbl_")
    else:
        graph = load_torch_to_networkx(root=root, name=dataset_name)

    experiment_params = get_training_params(config, 'training_params')



    embedding_model = HybridGraphEmbedding(
        graph,
        num_walks=experiment_params['num_walks'], 
        walk_length=experiment_params['walk_length'],
        num_landmarks = experiment_params['num_landmarks'],
        num_candidate_landmarks = experiment_params['num_candidate_landmarks'], 
        embedding_dim=experiment_params['embedding_dim'],
        learning_rate=experiment_params['learning_rate'],
        num_epochs=experiment_params['num_epochs'],
        workers=experiment_params['workers'],
        patience=experiment_params['patience'],
        alpha=experiment_params['alpha']
        )

    embedding_model.to(device)

    init_embeddings, walks = is_load(embedding_model,  dataset_name, load = False)
    
    by_landmark(embedding_model, init_embeddings, walks, dataset_name)





if __name__ == "__main__":
    if torch.cuda.is_available():
        device = f'cuda:0'  
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = 'cpu'
    device = torch.device(device)
    print("Device:", device)    

    main()
