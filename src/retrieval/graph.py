"""
Graph-based retriever — tracks cross-references between legal articles.
Uses NetworkX to store and traverse the relation graph.
"""
from __future__ import annotations
import re
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import networkx as nx
except ImportError:
    nx = None


class LegalGraph:
    """
    Simple knowledge graph tracking cross-references between legal articles.
    
    Nodes = article chunks (keyed by source:article)
    Edges = cross-references ("Điều X" mentioned in another article)
    """

    def __init__(self):
        if nx is None:
            raise ImportError("networkx is required for graph retrieval")
        self.graph = nx.DiGraph()

    def build_from_chunks(self, chunks: list[dict]) -> None:
        """Build graph from chunked documents by detecting cross-references."""
        # Add nodes
        for i, chunk in enumerate(chunks):
            node_id = self._chunk_id(chunk, i)
            self.graph.add_node(node_id, **chunk)

        # Add edges for cross-references
        chunk_ids = list(self.graph.nodes)
        for i, chunk in enumerate(chunks):
            src_id = self._chunk_id(chunk, i)
            refs = self._extract_references(chunk["page_content"])
            for ref_article in refs:
                # Find target node by article number
                for j, other_chunk in enumerate(chunks):
                    other_id = self._chunk_id(other_chunk, j)
                    if other_id == src_id:
                        continue
                    other_article = str(other_chunk.get("metadata", {}).get("article", ""))
                    if other_article == ref_article:
                        self.graph.add_edge(src_id, other_id, reference=f"Điều {ref_article}")

        print(f"  ✅ Legal graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def get_related(self, query_chunks: list[dict], max_hops: int = 2) -> list[dict]:
        """
        Given some query result chunks, find related chunks
        by traversing cross-references up to max_hops.
        """
        related = []
        visited = set()

        for chunk in query_chunks:
            # Find matching node
            for node_id, data in self.graph.nodes(data=True):
                if data.get("page_content", "")[:100] == chunk.get("page_content", "")[:100]:
                    self._bfs_collect(node_id, max_hops, visited, related)
                    break

        return related

    def _bfs_collect(self, start: str, max_hops: int, visited: set, result: list):
        """BFS from start node to collect related chunks."""
        queue = [(start, 0)]
        while queue:
            node, depth = queue.pop(0)
            if node in visited or depth > max_hops:
                continue
            visited.add(node)
            if depth > 0:  # Don't add the start node itself
                data = dict(self.graph.nodes[node])
                page_content = data.pop("page_content", "")
                metadata = data.pop("metadata", {})
                result.append({"page_content": page_content, "metadata": {**metadata, "graph_hop": depth}})

            for neighbor in self.graph.successors(node):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

    @staticmethod
    def _chunk_id(chunk: dict, index: int) -> str:
        meta = chunk.get("metadata", {})
        source = meta.get("filename", "unknown")
        article = meta.get("article", index)
        sub = meta.get("sub_chunk", 0)
        return f"{source}:Điều_{article}:{sub}"

    @staticmethod
    def _extract_references(text: str) -> list[str]:
        """Extract cross-referenced article numbers from text like 'Điều 29', 'khoản 1 Điều 36'."""
        # Match patterns like "Điều 29", "Điều 36", etc.
        matches = re.findall(r"(?:Điều|điều)\s+(\d+[a-zA-Z]?)", text)
        return list(set(matches))


# Module-level instance
_graph: LegalGraph | None = None


def get_graph() -> LegalGraph:
    global _graph
    if _graph is None:
        _graph = LegalGraph()
    return _graph


def build_graph(chunks: list[dict]) -> None:
    graph = get_graph()
    graph.build_from_chunks(chunks)
