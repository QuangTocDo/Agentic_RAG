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
    
    Nodes = article chunks (keyed by unique chunk_id)
    Edges = cross-references with relationship types (amendment, repeal, guidance, reference)
    """

    def __init__(self):
        if nx is None:
            raise ImportError("networkx is required for graph retrieval")
        self.graph = nx.DiGraph()

    def build_from_chunks(self, chunks: list[dict]) -> None:
        """Build graph from chunked documents by detecting cross-references in O(N)."""
        self.graph.clear()
        
        # Add all nodes first
        for i, chunk in enumerate(chunks):
            node_id = self._chunk_id(chunk, i)
            self.graph.add_node(
                node_id,
                page_content=chunk.get("page_content", ""),
                metadata=chunk.get("metadata", {})
            )

        # Build lookup table: (doc_key, article_id) -> list of node_ids
        lookup = {}
        for node_id, data in self.graph.nodes(data=True):
            meta = data.get("metadata", {})
            doc_key = meta.get("doc_id", meta.get("filename", ""))
            article = str(meta.get("article", ""))
            if doc_key and article:
                key = (doc_key, article)
                if key not in lookup:
                    lookup[key] = []
                lookup[key].append(node_id)

        # Create edges using the lookup table (only within the same document)
        for node_id, data in self.graph.nodes(data=True):
            content = data.get("page_content", "")
            meta = data.get("metadata", {})
            doc_key = meta.get("doc_id", meta.get("filename", ""))
            
            refs = self._extract_references(content)
            for ref_article in refs:
                target_nodes = lookup.get((doc_key, ref_article), [])
                for target_node in target_nodes:
                    if target_node != node_id:
                        rel = self._classify_relationship(content, ref_article)
                        self.graph.add_edge(node_id, target_node, rel_type=rel, label=f"Điều {ref_article}")

        print(f"  ✅ Legal graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def append_from_chunks(self, new_chunks: list[dict]) -> None:
        """Load existing graph, append new chunks, detect references in O(M)."""
        try:
            self.load()
        except FileNotFoundError:
            self.graph = nx.DiGraph()

        new_node_ids = []
        for i, chunk in enumerate(new_chunks):
            node_id = self._chunk_id(chunk, i)
            self.graph.add_node(
                node_id,
                page_content=chunk.get("page_content", ""),
                metadata=chunk.get("metadata", {})
            )
            new_node_ids.append(node_id)

        # Build lookup table for the entire graph
        lookup = {}
        for node_id, data in self.graph.nodes(data=True):
            meta = data.get("metadata", {})
            doc_key = meta.get("doc_id", meta.get("filename", ""))
            article = str(meta.get("article", ""))
            if doc_key and article:
                key = (doc_key, article)
                if key not in lookup:
                    lookup[key] = []
                lookup[key].append(node_id)

        # Create edges for the new nodes only
        for node_id in new_node_ids:
            data = self.graph.nodes[node_id]
            content = data.get("page_content", "")
            meta = data.get("metadata", {})
            doc_key = meta.get("doc_id", meta.get("filename", ""))
            
            refs = self._extract_references(content)
            for ref_article in refs:
                target_nodes = lookup.get((doc_key, ref_article), [])
                for target_node in target_nodes:
                    if target_node != node_id:
                        rel = self._classify_relationship(content, ref_article)
                        self.graph.add_edge(node_id, target_node, rel_type=rel, label=f"Điều {ref_article}")

        print(f"  ✅ Legal graph updated: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

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

    def get_related(self, query_chunks: list[dict], max_hops: int = 2) -> list[dict]:
        """
        Given query result chunks, find related chunks by traversing
        references up to max_hops using their unique chunk_id.
        """
        related = []
        visited = set()

        for chunk in query_chunks:
            meta = chunk.get("metadata", {})
            chunk_id = meta.get("chunk_id")
            if chunk_id and chunk_id in self.graph:
                self._bfs_collect(chunk_id, max_hops, visited, related)

        return related

    def _bfs_collect(self, start: str, max_hops: int, visited: set, result: list):
        """BFS from start node using collections.deque and direction weights."""
        from collections import deque
        
        # queue stores (node, depth, score)
        queue = deque([(start, 0, 1.0)])
        while queue:
            node, depth, score = queue.popleft()
            if node in visited or depth > max_hops or score < 0.2:
                continue
            
            visited.add(node)
            if depth > 0:
                data = dict(self.graph.nodes[node])
                page_content = data.pop("page_content", "")
                metadata = data.pop("metadata", {})
                result.append({
                    "page_content": page_content,
                    "metadata": {
                        **metadata,
                        "graph_hop": depth,
                        "graph_score": score
                    }
                })

            # Outgoing references (successors) - direct dependencies, weight = 1.0
            for neighbor in self.graph.successors(node):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1, score * 0.8))

            # Incoming references (predecessors) - indirect dependencies, weight = 0.5
            for neighbor in self.graph.predecessors(node):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1, score * 0.4))

    @staticmethod
    def _classify_relationship(text: str, article_num: str) -> str:
        """Classify reference type based on keywords surrounding it."""
        pattern = re.compile(rf"(?:sửa đổi|bổ sung|bãi bỏ|hướng dẫn|áp dụng|quy định)\s+[^.]*?Điều\s+{article_num}", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            matched_text = match.group(0).lower()
            if "sửa đổi" in matched_text or "bổ sung" in matched_text:
                return "amendment"
            if "bãi bỏ" in matched_text:
                return "repeal"
            if "hướng dẫn" in matched_text:
                return "guidance"
        return "reference"

    @staticmethod
    def _chunk_id(chunk: dict, index: int) -> str:
        meta = chunk.get("metadata", {})
        if "chunk_id" in meta:
            return meta["chunk_id"]
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


def graph_search(vector_store, graph_instance, query: str, k: int, initial_k: int, max_hops: int) -> list[dict]:
    """
    Search documents using graph-guided multi-hop retrieval.
    1. Search vector_store to get initial_k seeds.
    2. Traverse graph_instance to expand from seeds up to max_hops.
    3. Return deduplicated combination of seeds and expanded candidates, capped at k.
    """
    from src.retrieval.dense import dense_search
    seeds = dense_search(query, k=initial_k)
    expanded = graph_instance.get_related(seeds, max_hops=max_hops)
    from src.retrieval.hybrid import _deduplicate
    combined = _deduplicate(seeds + expanded)
    return combined[:k]
