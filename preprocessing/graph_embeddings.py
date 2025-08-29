import pandas as pd
import networkx as nx
import numpy as np
from node2vec import Node2Vec
import os


class GraphEmbedder:
    def __init__(self, graph_df, dimensions=8, walk_length=10, num_walks=50,
                 workers=2, p=1, q=1):
        """
        graph_df: DataFrame with 'Username'
        dimensions: embedding vector size
        walk_length: length of each random walk
        num_walks: number of walks per node
        p, q: Node2Vec return/in-out parameters for walk bias
        """
        self.graph_df = graph_df
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.p = p
        self.q = q
        self.graph = None
        self.embeddings = None

    def build_graph(self):
        """Simulate a social/payment network with clustering + varied weights"""
        self.graph = nx.Graph()

        # Add nodes
        for user in self.graph_df['Username']:
            self.graph.add_node(user)

        # Randomly connect users to simulate transactions/social ties
        users = list(self.graph_df['Username'])
        np.random.seed(42)  # reproducibility

        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                # 🔹 Higher connection chance for users with nearby indices (clusters)
                base_prob = 0.20
                if abs(i - j) < 4:  # local neighborhood gets extra edges
                    prob = base_prob + 0.25
                else:
                    prob = base_prob

                if np.random.rand() < prob:
                    # 🔹 Stronger + more varied weights (0.5–5.0)
                    weight = round(np.random.uniform(0.5, 5.0), 2)
                    self.graph.add_edge(users[i], users[j], weight=weight)

        print(f"[INFO] Graph built with {self.graph.number_of_nodes()} nodes "
              f"and {self.graph.number_of_edges()} edges")
        return self

    def generate_node_embeddings(self):
        """Run Node2Vec on the simulated graph"""
        node2vec = Node2Vec(
            self.graph, dimensions=self.dimensions,
            walk_length=self.walk_length, num_walks=self.num_walks,
            workers=self.workers, p=self.p, q=self.q
        )
        model = node2vec.fit(window=5, min_count=1, batch_words=4)

        # Embeddings aligned with usernames
        self.embeddings = np.array([
            model.wv[str(node)] if str(node) in model.wv else np.zeros(self.dimensions)
            for node in self.graph_df['Username']
        ])
        print(f"[INFO] Generated node embeddings of shape {self.embeddings.shape}")
        return self.embeddings


if __name__ == "__main__":
    # 🔹 Select whether to run on training or production dataset
    dataset_type = "training"  # change to "production" if testing prod data
    dataset_path = os.path.join("data", dataset_type, "neural_credit_data.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"[ERROR] Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    graph_df = df[['Username']]

    embedder = GraphEmbedder(graph_df)
    embedder.build_graph()
    embeddings = embedder.generate_node_embeddings()

    print("[DEBUG] First 2 node embeddings:\n", embeddings[:2])

