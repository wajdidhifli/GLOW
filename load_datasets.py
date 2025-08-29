import networkx as nx
from torch_geometric.datasets import Planetoid, Coauthor,CitationFull,AttributedGraphDataset
import torch_geometric.transforms as T
import yaml 
from joblib import Parallel, delayed
from tqdm import tqdm
from ogb.linkproppred import LinkPropPredDataset, PygLinkPropPredDataset
import random
import math



def get_training_params(config, params='training_params'):

    return config[params]

def load_config(file_path='config.yaml'):

    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def load_torch_to_networkx(root='./data', name="cora"):

    if name in ['cora', 'citeseer', 'pubmed']:

        dataset = Planetoid(root=root, name=name, transform=T.NormalizeFeatures())
    elif name in ["physics", "cs"]:
        dataset = Coauthor(root=root, name=name, transform=T.NormalizeFeatures())
    elif name=="dblp":
        dataset = CitationFull(root=root, name=name , transform=T.NormalizeFeatures())
    elif name=="blogcatalog":
        dataset = AttributedGraphDataset(root=root, name=name , transform=T.NormalizeFeatures())
    

    data = dataset[0]
    print(data)

    G = nx.Graph()
    

    for i in range(data.x.shape[0]):
        node_features = data.x[i].numpy()  
        G.add_node(i, features=node_features)

    for i in range(data.edge_index.shape[1]):
        node1 = data.edge_index[0, i].item()  
        node2 = data.edge_index[1, i].item()  
        G.add_edge(node1, node2)
    
    for i in range(data.y.shape[0]):
        label = data.y[i].item() 
        G.nodes[i]['label'] = label
    

    connected_components = list(nx.connected_components(G))



    num_components = len(connected_components)

    largest_component = max(connected_components, key=len)


    G_largest = G.subgraph(largest_component)

    G.remove_nodes_from(list(nx.isolates(G)))

    V = G.number_of_nodes()
    E = G.number_of_edges()

    max_edges = V * (V - 1) / 2

    density = E / max_edges

    print(f"Number of nodes (V): {V}")
    print(f"Number of edges (E): {E}")
    print(f"Maximum possible edges: {max_edges}")
    print(f"Edge Density: {density}")


    if density > 0.5:
        print("The graph is dense.")
    else:
        print("The graph is sparse.")



    edge_density = nx.density(G_largest)
    print(f"Edge Density: {edge_density}")
    return G


def load_dgl_to_networkx(root='./data', name="cora"):

    from dgl import data
    if name=="cora":
        dataset = data.CoraGraphDataset(raw_dir = root)
    elif name=="citeseer":
        dataset = data.CiteseerGraphDataset(raw_dir = root)
    elif name=="pubmed":
        dataset = data.PubmedGraphDataset(raw_dir = root)
    elif name=="cs":
        dataset = data.CoauthorCSDataset(raw_dir = root)
    elif name=="physics":
        dataset = data.CoauthorPhysicsDataset(raw_dir = root)
    data = dataset[0]
    num_class = dataset.num_classes

    train_mask = val_mask = test_mask = None
    if name in ['cora', 'citeseer', 'pubmed']:

        train_mask = data.ndata['train_mask']
        val_mask = data.ndata['val_mask']
        test_mask = data.ndata['test_mask']

    label = data.ndata['label']



    G = nx.Graph()
    

    for i in range(data.x.shape[0]):
        node_features = data.x[i].numpy() 
        G.add_node(i, features=node_features)
    

    for i in range(data.edge_index.shape[1]):
        node1 = data.edge_index[0, i].item()  
        node2 = data.edge_index[1, i].item()  
        G.add_edge(node1, node2)
    

    for i in range(data.y.shape[0]):
        label = data.y[i].item()  # Convert tensor to scalar
        G.nodes[i]['label'] = label

    connected_components = list(nx.connected_components(G))


    num_components = len(connected_components)

    largest_component = max(connected_components, key=len)

    G_largest = G.subgraph(largest_component)
    print(f'Number of connected components: {num_components}, {len(G_largest.nodes)}')

   
    print(f"Number of isolated nodes: {len(list(nx.isolates(G)))}")
    G.remove_nodes_from(list(nx.isolates(G)))


    V = G.number_of_nodes()
    E = G.number_of_edges()

    max_edges = V * (V - 1) / 2

    density = E / max_edges


    print(f"Number of nodes (V): {V}")
    print(f"Number of edges (E): {E}")
    print(f"Maximum possible edges: {max_edges}")
    print(f"Edge Density: {density}")


    if density > 0.5:
        print("The graph is dense.")
    else:
        print("The graph is sparse.")



    edge_density = nx.density(G_largest)
    print(f"Edge Density: {edge_density}")
    return G, (train_mask, val_mask, test_mask)

def load_ogbl_to_networkx(root='./data/', name='ogbl-collab'):

    dataset = LinkPropPredDataset(name=name, root=root)
    graph = dataset[0]  


    G = nx.Graph()

    num_nodes = graph['num_nodes']
    G.add_nodes_from(range(num_nodes))

    edges = graph['edge_index'].T  
    G.add_edges_from(edges.tolist())

    

    connected_components = list(nx.connected_components(G))


    num_components = len(connected_components)

    largest_component = max(connected_components, key=len)

    G_largest = G.subgraph(largest_component)
    print(f'Number of connected components: {num_components}, {len(G_largest.nodes)}')
    
    print(f"Number of isolated nodes: {len(list(nx.isolates(G)))}")
    G.remove_nodes_from(list(nx.isolates(G)))

    V = G.number_of_nodes()
    E = G.number_of_edges()

    max_edges = V * (V - 1) / 2

    density = E / max_edges


    print(f"Number of nodes (V): {V}")
    print(f"Number of edges (E): {E}")
    print(f"Maximum possible edges: {max_edges}")
    print(f"Edge Density: {density}")


    if density > 0.5:
        print("The graph is dense.")
    else:
        print("The graph is sparse.")
    


    edge_density = nx.density(G)
    print(f"Edge Density: {edge_density}")

    return G

def load_ogbl_train_val_test(root='./data/', name='ogbl-collab'):

    dataset = PygLinkPropPredDataset(name = name) 

    split_edge = dataset.get_edge_split()
    train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]

    return train_edge, valid_edge, test_edge

def compute_shortest_path_length(graph, node):
    return nx.single_source_shortest_path_length(graph, node)



def remove_small_components(G, walk_length, connected_components):

    i = 0
    

    for component in connected_components:
        if len(component) < walk_length:

            G.remove_nodes_from(component)
            i += 1

    if i > 0:
        print(f"Removed {i} small components with size < {walk_length}. Remaining total nb of nodes: {G.number_of_nodes()}")


    return G

def bfs_farthest_node(graph, start_node):

    visited = {start_node: 0}
    queue = [start_node]
    farthest_node = start_node
    max_distance = 0

    while queue:
        current_node = queue.pop(0)
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                visited[neighbor] = visited[current_node] + 1
                queue.append(neighbor)
                if visited[neighbor] > max_distance:
                    max_distance = visited[neighbor]
                    farthest_node = neighbor

    return farthest_node, max_distance

def approximate_diameter_largest_component(graph, num_samples=100, num_jobs=-1):

    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise TypeError("Input 'graph' must be a NetworkX graph object.")


    connected_components = list(nx.connected_components(graph))


    if not connected_components:
        return [], 0, []


    largest_component = max(connected_components, key=len)
    subgraph = graph.subgraph(largest_component).copy()

  
    nodes = list(subgraph.nodes)
    if len(nodes) <= num_samples:
        sampled_nodes = nodes  
    else:
        sampled_nodes = random.sample(nodes, num_samples)


    results = Parallel(n_jobs=num_jobs)(
        delayed(bfs_farthest_node)(subgraph, node) for node in tqdm(sampled_nodes, desc="Computing BFS")
    )


    max_distance = max(result[1] for result in results)

    return connected_components, max_distance


def adaptive_sampling_diameter(graph, initial_samples=None, max_samples=10000, tolerance=0.01, num_jobs=-1):

    connected_components = list(nx.connected_components(graph))
    

    largest_component = max(connected_components, key=len)
    subgraph = graph.subgraph(largest_component).copy()
    N = len(largest_component)

    if initial_samples is None:
        initial_samples = min(1000, int(math.sqrt(N)))

    num_samples = initial_samples
    prev_diameter = 0
    nodes = list(subgraph.nodes)

    while num_samples <= max_samples:

        sampled_nodes = random.sample(nodes, min(num_samples, len(nodes)))

        results = Parallel(n_jobs=num_jobs)(
            delayed(bfs_farthest_node)(subgraph, node) for node in tqdm(sampled_nodes, desc=f"Computing BFS with {num_samples} samples")
        )

        current_diameter = max(result[1] for result in results)


        if prev_diameter > 0 and abs(current_diameter - prev_diameter) / prev_diameter < tolerance:
            break

        prev_diameter = current_diameter
        num_samples *= 2  

    return connected_components, current_diameter


def degree_based_sampling_diameter(graph, num_samples=None, num_jobs=-1):

    connected_components = list(nx.connected_components(graph))
    
    largest_component = max(connected_components, key=len)
    subgraph = graph.subgraph(largest_component).copy()
    N = len(largest_component)

    # Set num_samples based on min(1000, sqrt(N))
    if num_samples is None:
        num_samples = min(1000, int(math.sqrt(N)))

    degrees = dict(subgraph.degree())

    sampled_nodes = random.choices(list(subgraph.nodes), weights=degrees.values(), k=num_samples)

    results = Parallel(n_jobs=num_jobs)(
        delayed(bfs_farthest_node)(subgraph, node) for node in tqdm(sampled_nodes, desc="Computing BFS with degree-based sampling")
    )

    max_distance = max(result[1] for result in results)

    return connected_components, max_distance

if __name__ == "__main__":
    

    
    G = load_torch_to_networkx(name="pubmed")


    _, approx_diameter = approximate_diameter_largest_component(G, num_samples=100, num_jobs=-1)
    print(f"Approximate diameter of the graph: {approx_diameter}")
    exit()


    _, diameter = adaptive_sampling_diameter(G, initial_samples=100, max_samples=10000, tolerance=0.01, num_jobs=-1)
    print(f"Approximate diameter (adaptive sampling): {diameter}")



    _, diameter = degree_based_sampling_diameter(G, num_samples=100, num_jobs=-1)
    print(f"Approximate diameter (degree-based sampling): {diameter}")


    