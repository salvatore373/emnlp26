import json
import pickle
from typing import Any

import networkx as nx

from tools.base_noisy_tool import BaseNoisyTool

# DBLP Description to include in docstrings
DBLP_DOCSTRING = """
# The DBLP Networks Explained
The DBLP database maps the evolution of computer science research through two interconnected networks. Both are extracted from an underlying **bipartite graph** that links authors directly to the papers they have written.

## 1. The Paper Citation Network
Maps the flow of knowledge and identifies influential research.
* **Nodes:** Published papers.
* **Edges:** Directed (Arrows). An edge points from Paper A to Paper B if A cites B.

## 2. The Author Collaboration Network
Maps the social and professional relationships between researchers.
* **Nodes:** Individual authors.
* **Edges:** Undirected (Lines). A line connects two authors if they have co-authored a paper. This line is often weighted (thicker) based on the number of joint publications.
"""

DBLP_DATA_PATH = "/workspace/tools/toolqa_preprocessing/data/dblp"


class DblpTools:
    """
    A wrapper class that encapsulates all shared helpers and DBLP tools.
    """
    paper_net = None
    author_net = None
    id2title_dict = None
    title2id_dict = None
    id2author_dict = None
    author2id_dict = None
    _loaded = False

    @classmethod
    def _load_graph(cls, path: str):
        if cls._loaded:
            return

        try:
            with open(f'{path}/paper_net.pkl', 'rb') as f:
                cls.paper_net = pickle.load(f)
            with open(f'{path}/author_net.pkl', 'rb') as f:
                cls.author_net = pickle.load(f)
            with open(f"{path}/title2id_dict.pkl", "rb") as f:
                cls.title2id_dict = pickle.load(f)
            with open(f"{path}/author2id_dict.pkl", "rb") as f:
                cls.author2id_dict = pickle.load(f)
            with open(f"{path}/id2title_dict.pkl", "rb") as f:
                cls.id2title_dict = pickle.load(f)
            with open(f"{path}/id2author_dict.pkl", "rb") as f:
                cls.id2author_dict = pickle.load(f)
            cls._loaded = True
        except FileNotFoundError as e:
            # Create dummy empty graphs and dictionaries for fail case testing if data isn't present
            cls.paper_net = nx.DiGraph()
            cls.author_net = nx.Graph()
            cls.id2title_dict = {}
            cls.title2id_dict = {}
            cls.id2author_dict = {}
            cls.author2id_dict = {}
            cls._loaded = True

    @classmethod
    def check_neighbours(cls, path: str, argument: str):
        cls._load_graph(path)
        try:
            graph_type, node = argument.split(', ')
            if graph_type == 'PaperNet':
                graph = cls.paper_net
                dictionary = cls.title2id_dict
                inv_dict = cls.id2title_dict
                entity_type = "Paper"
            elif graph_type == 'AuthorNet':
                graph = cls.author_net
                dictionary = cls.author2id_dict
                inv_dict = cls.id2author_dict
                entity_type = "Author"
            else:
                return [{"error": f"Unknown network type: {graph_type}"}]

            if node not in dictionary:
                return [{"error": f"{entity_type} '{node}' is not a node of the graph."}]

            node_id = dictionary[node]
            if node_id not in graph:
                return [{"error": f"{entity_type} '{node}' is not a node of the graph."}]

            neighbour_list = []
            # For directed graphs (PaperNet), include both incoming and outgoing neighbors
            if isinstance(graph, nx.DiGraph):
                neighbours = set()
                neighbours.update(graph.predecessors(node_id))
                neighbours.update(graph.successors(node_id))
            else:
                neighbours = graph.neighbors(node_id)

            for neighbour in neighbours:
                neighbour_list.append(inv_dict.get(neighbour, neighbour))
            return neighbour_list
        except Exception as e:
            return [{"error": str(e)}]

    @classmethod
    def check_nodes(cls, path: str, argument: str):
        cls._load_graph(path)
        try:
            graph_type, node = argument.split(', ')
            if graph_type == 'PaperNet':
                graph = cls.paper_net
                dictionary = cls.title2id_dict
                inv_dict = cls.id2title_dict
                entity_type = "Paper"
            elif graph_type == 'AuthorNet':
                graph = cls.author_net
                dictionary = cls.author2id_dict
                inv_dict = cls.id2author_dict
                entity_type = "Author"
            else:
                return {"error": f"Unknown network type: {graph_type}"}

            if node not in dictionary:
                return {"error": f"{entity_type} '{node}' is not a node of the graph."}

            node_id = dictionary[node]
            if node_id not in graph:
                return {"error": f"{entity_type} '{node}' is not a node of the graph."}

            node = graph.nodes[node_id]

            if 'n_citation' in node:
                node['n_citation'] = int(node['n_citation'])
            if 'year' in node:
                node['year'] = int(node['year'])

            return node
        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def check_edges(cls, path: str, argument: str):
        cls._load_graph(path)
        try:
            graph_type, node1, node2 = argument.split(', ')
            if graph_type == 'PaperNet':
                graph = cls.paper_net
                dictionary = cls.title2id_dict
                inv_dict = cls.id2title_dict
                entity_type = "Paper"
            elif graph_type == 'AuthorNet':
                graph = cls.author_net
                dictionary = cls.author2id_dict
                inv_dict = cls.id2title_dict  # In reference code, edges contain paper list mapped to titles
                entity_type = "Author"
            else:
                return {"error": f"Unknown network type: {graph_type}"}

            if node1 not in dictionary or node2 not in dictionary:
                return {"error": "One or both elements are not nodes of the graph."}

            node1_id = dictionary[node1]
            node2_id = dictionary[node2]

            if node1_id not in graph or node2_id not in graph:
                return {"error": "One or both elements are not nodes of the graph."}

            if not graph.has_edge(node1_id, node2_id):
                return {"error": f"An edge between {entity_type} '{node1}' and '{node2}' does not exist."}

            edge = graph.edges[node1_id, node2_id]

            # Translate paper ids to titles for AuthorNet
            if graph_type == 'AuthorNet' and 'papers' in edge:
                papers_list = []
                for p_id in edge['papers']:
                    if p_id in inv_dict:
                        papers_list.append(inv_dict[p_id])
                    else:
                        papers_list.append(p_id)
                edge['papers'] = papers_list

            if 'n_citation' in edge:
                edge['n_citation'] = [int(c) for c in edge['n_citation']]

            return edge
        except Exception as e:
            return {"error": str(e)}

    # ==========================================
    # Inner Tool Classes
    # ==========================================

    class GetDblpAuthorNodeTool(BaseNoisyTool):
        name = "get_dblp_author_node"
        description = (
                "Returns the JSON object representing an author and the information stored in the DBLP citation network.\n" + DBLP_DOCSTRING
        )
        output_type = "string"
        inputs = {
            "author_name": {"type": "string", "description": "The name of the author to retrieve."}
        }

        def execute_tool(self, author_name: str) -> Any:
            res = DblpTools.check_nodes(DBLP_DATA_PATH, f'AuthorNet, {author_name}')
            return json.dumps(res, indent=2)

    class GetDblpAuthorNeighborsTool(BaseNoisyTool):
        name = "get_dblp_author_neighbors"
        description = (
                "Retrieves all the given author's collaborators according to DBLP citation network, "
                "i.e. authors that appear as co-authors of author_name in at least one paper.\n" + DBLP_DOCSTRING
        )
        output_type = "string"
        inputs = {
            "author_name": {"type": "string", "description": "The name of the author to retrieve collaborators for."}
        }

        def execute_tool(self, author_name: str) -> Any:
            res = DblpTools.check_neighbours(DBLP_DATA_PATH, f'AuthorNet, {author_name}')
            return json.dumps(res, indent=2)

    class GetDblpAuthorEdgesTool(BaseNoisyTool):
        name = "get_dblp_author_edges"
        description = (
                "Returns the edge connecting two authors, giving information about their collaborations.\n" + DBLP_DOCSTRING
        )
        output_type = "string"
        inputs = {
            "author1": {"type": "string", "description": "The first author's name."},
            "author2": {"type": "string", "description": "The second author's name."}
        }

        def execute_tool(self, author1: str, author2: str) -> Any:
            res = DblpTools.check_edges(DBLP_DATA_PATH, f'AuthorNet, {author1}, {author2}')
            return json.dumps(res, indent=2)

    class GetDblpPaperNodeTool(BaseNoisyTool):
        name = "get_dblp_paper_node"
        description = (
                "Returns the JSON object representing a paper and the information stored in the DBLP citation network.\n" + DBLP_DOCSTRING
        )
        output_type = "string"
        inputs = {
            "paper_name": {"type": "string", "description": "The name of the paper to retrieve."}
        }

        def execute_tool(self, paper_name: str) -> Any:
            res = DblpTools.check_nodes(DBLP_DATA_PATH, f'PaperNet, {paper_name}')
            return json.dumps(res, indent=2)

    class GetDblpPaperNeighborsTool(BaseNoisyTool):
        name = "get_dblp_paper_neighbors"
        description = (
                "Returns the neighbors of a node's paper in the DBLP citation network.\n" + DBLP_DOCSTRING
        )
        output_type = "string"
        inputs = {
            "paper_name": {"type": "string", "description": "The name of the paper to retrieve neighbors for."}
        }

        def execute_tool(self, paper_name: str) -> Any:
            res = DblpTools.check_neighbours(DBLP_DATA_PATH, f'PaperNet, {paper_name}')
            return json.dumps(res, indent=2)

    class GetDblpPaperEdgesTool(BaseNoisyTool):
        name = "get_dblp_paper_edges"
        description = (
                "Returns the edges between two paper nodes in the DBLP citation network.\n" + DBLP_DOCSTRING
        )
        output_type = "string"
        inputs = {
            "paper_name_1": {"type": "string", "description": "The first paper's name."},
            "paper_name_2": {"type": "string", "description": "The second paper's name."}
        }

        def execute_tool(self, paper_name_1: str, paper_name_2: str) -> Any:
            res = DblpTools.check_edges(DBLP_DATA_PATH, f'PaperNet, {paper_name_1}, {paper_name_2}')

            res = json.dumps(res, indent=2)
            if res == '{}':
                res += ' (An edge with no metadata exists)'
            return res
