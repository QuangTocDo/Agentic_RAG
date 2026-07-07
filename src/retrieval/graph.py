"""
Graph-based retriever — tracks cross-references between legal articles.
Uses NetworkX to store and traverse the relation graph.
"""
from __future__ import annotations
import pickle
import re
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.setting import settings

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
        self.content_to_node = {}

    def build_from_chunks(self, chunks: list[dict]) -> None:
        """Build graph from chunked documents by detecting cross-references."""
        self.graph.clear()
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
        self._rebuild_lookup_cache()

    def append_from_chunks(self, new_chunks: list[dict]) -> None:
        """Load existing graph, merge new chunks, detect cross-references, and save."""
        try:
            self.load()
        except FileNotFoundError:
            self.graph = nx.DiGraph()

        # Collect existing chunks stored in graph nodes
        existing_chunks = []
        for node_id, data in self.graph.nodes(data=True):
            # Reconstruct the chunk dict
            existing_chunks.append({
                "page_content": data.get("page_content", ""),
                "metadata": data.get("metadata", {})
            })

        # Deduplicate
        existing_contents = {c["page_content"] for c in existing_chunks}
        for chunk in new_chunks:
            if chunk["page_content"] not in existing_contents:
                existing_chunks.append(chunk)
                existing_contents.add(chunk["page_content"])

        # Re-build the full graph from the merged set of chunks
        self.build_from_chunks(existing_chunks)


    def save(self, path: str | None = None) -> None:
        """Persist the graph to disk."""
        if path is None:
            path = settings.graph_index_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"  ✅ Legal graph saved to {path}")

    def load(self, path: str | None = None) -> None:
        """Load a persisted graph from disk."""
        if path is None:
            path = settings.graph_index_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Legal graph not found at: {path}")
        with open(path, "rb") as f:
            self.graph = pickle.load(f)
        print(f"  ✅ Legal graph loaded ({self.graph.number_of_nodes()} nodes)")
        self._rebuild_lookup_cache()

    def _rebuild_lookup_cache(self) -> None:
        """Rebuild the page_content -> node_id lookup dictionary."""
        self.content_to_node = {}
        for node_id, data in self.graph.nodes(data=True):
            content_key = data.get("page_content", "")[:100]
            self.content_to_node[content_key] = node_id

    def get_related(self, query_chunks: list[dict], max_hops: int = 2) -> list[dict]:
        """
        Given some query result chunks, find related chunks
        by traversing cross-references up to max_hops.
        """
        if not hasattr(self, "content_to_node") or not self.content_to_node:
            self._rebuild_lookup_cache()

        related = []
        visited = set()

        for chunk in query_chunks:
            key = chunk.get("page_content", "")[:100]
            node_id = self.content_to_node.get(key)
            if node_id:
                self._bfs_collect(node_id, max_hops, visited, related)

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
    graph.save()


def append_graph(chunks: list[dict]) -> None:
    graph = get_graph()
    graph.append_from_chunks(chunks)
    graph.save()



def load_graph(path: str | None = None) -> LegalGraph:
    graph = get_graph()
    if graph.graph.number_of_nodes() == 0:
        graph.load(path)
    return graph
