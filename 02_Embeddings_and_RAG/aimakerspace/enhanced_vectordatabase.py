import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Any, Optional
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio
import json


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the euclidean distance between two vectors (lower is more similar)."""
    return np.linalg.norm(vector_a - vector_b)


def manhattan_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Manhattan distance between two vectors (lower is more similar)."""
    return np.sum(np.abs(vector_a - vector_b))


def dot_product_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the dot product similarity between two vectors."""
    return np.dot(vector_a, vector_b)


class EnhancedVectorDatabase:
    """
    Enhanced Vector Database with metadata support and multiple distance metrics.
    """
    
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.metadata = defaultdict(dict)  # Store metadata for each document
        self.embedding_model = embedding_model or EmbeddingModel()
        
        # Available distance metrics
        self.distance_metrics = {
            "cosine": cosine_similarity,
            "euclidean": euclidean_distance,
            "manhattan": manhattan_distance,
            "dot_product": dot_product_similarity
        }

    def insert(self, key: str, vector: np.array, metadata: Dict[str, Any] = None) -> None:
        """Insert a vector with optional metadata."""
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: str = "cosine",
        metadata_filter: Dict[str, Any] = None,
        return_scores: bool = True,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors with optional metadata filtering.
        
        Args:
            query_vector: The query vector to search for
            k: Number of results to return
            distance_measure: Distance metric to use ("cosine", "euclidean", "manhattan", "dot_product")
            metadata_filter: Optional dictionary to filter results by metadata
            return_scores: Whether to return similarity scores
            
        Returns:
            List of tuples containing (text, score, metadata)
        """
        if distance_measure not in self.distance_metrics:
            raise ValueError(f"Unknown distance measure: {distance_measure}. Available: {list(self.distance_metrics.keys())}")
        
        distance_func = self.distance_metrics[distance_measure]
        
        # Calculate scores for all vectors
        scores = []
        for key, vector in self.vectors.items():
            # Apply metadata filtering if specified
            if metadata_filter and not self._matches_filter(key, metadata_filter):
                continue
                
            score = distance_func(query_vector, vector)
            metadata = self.metadata.get(key, {})
            scores.append((key, score, metadata))
        
        # Sort based on distance metric (cosine and dot_product: higher is better, others: lower is better)
        reverse_sort = distance_measure in ["cosine", "dot_product"]
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=reverse_sort)
        
        # Return top k results
        results = sorted_scores[:k]
        
        if return_scores:
            return results
        else:
            return [(text, metadata) for text, _, metadata in results]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: str = "cosine",
        metadata_filter: Dict[str, Any] = None,
        return_as_text: bool = False,
        return_scores: bool = True,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search by text query with metadata support.
        """
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(
            query_vector, 
            k, 
            distance_measure, 
            metadata_filter, 
            return_scores
        )
        
        if return_as_text:
            return [result[0] for result in results]
        
        return results

    def _matches_filter(self, key: str, metadata_filter: Dict[str, Any]) -> bool:
        """Check if a document's metadata matches the filter criteria."""
        doc_metadata = self.metadata.get(key, {})
        
        for filter_key, filter_value in metadata_filter.items():
            if filter_key not in doc_metadata:
                return False
            
            doc_value = doc_metadata[filter_key]
            
            # Handle different types of filtering
            if isinstance(filter_value, dict):
                # Range filtering for numeric values
                if "$gte" in filter_value and doc_value < filter_value["$gte"]:
                    return False
                if "$lte" in filter_value and doc_value > filter_value["$lte"]:
                    return False
                if "$gt" in filter_value and doc_value <= filter_value["$gt"]:
                    return False
                if "$lt" in filter_value and doc_value >= filter_value["$lt"]:
                    return False
                if "$in" in filter_value and doc_value not in filter_value["$in"]:
                    return False
                if "$nin" in filter_value and doc_value in filter_value["$nin"]:
                    return False
            elif isinstance(filter_value, list):
                # Check if doc_value is in the list
                if doc_value not in filter_value:
                    return False
            else:
                # Exact match
                if doc_value != filter_value:
                    return False
        
        return True

    def retrieve_from_key(self, key: str) -> Tuple[np.array, Dict[str, Any]]:
        """Retrieve vector and metadata for a given key."""
        vector = self.vectors.get(key, None)
        metadata = self.metadata.get(key, {})
        return vector, metadata

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored metadata."""
        return dict(self.metadata)

    def update_metadata(self, key: str, new_metadata: Dict[str, Any]) -> None:
        """Update metadata for a specific document."""
        if key in self.vectors:
            self.metadata[key].update(new_metadata)
        else:
            raise KeyError(f"Key '{key}' not found in database")

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "total_documents": len(self.vectors),
            "total_metadata_entries": len(self.metadata),
            "vector_dimension": len(next(iter(self.vectors.values()))) if self.vectors else 0,
            "available_distance_metrics": list(self.distance_metrics.keys())
        }

    async def abuild_from_list(self, list_of_text: List[str], metadata_list: List[Dict[str, Any]] = None) -> "EnhancedVectorDatabase":
        """Build database from list of texts with optional metadata."""
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        metadata_list = metadata_list or [{}] * len(list_of_text)
        
        for text, embedding, metadata in zip(list_of_text, embeddings, metadata_list):
            self.insert(text, np.array(embedding), metadata)
        
        return self

    async def abuild_from_chunks(self, chunks: List[Dict[str, Any]]) -> "EnhancedVectorDatabase":
        """
        Build database from chunks that contain both text and metadata.
        Expected format: [{"text": "...", "metadata": {...}}, ...]
        
        This method ensures ALL metadata from chunks is preserved.
        """
        texts = [chunk["text"] for chunk in chunks]
        # Ensure we get complete metadata, not just references
        metadata_list = []
        for chunk in chunks:
            # Create a deep copy of metadata to ensure all fields are preserved
            chunk_metadata = chunk.get("metadata", {})
            if isinstance(chunk_metadata, dict):
                # Ensure all metadata fields are properly copied
                preserved_metadata = {}
                for key, value in chunk_metadata.items():
                    preserved_metadata[key] = value
                metadata_list.append(preserved_metadata)
            else:
                metadata_list.append({})
        
        return await self.abuild_from_list(texts, metadata_list)

    def save_to_file(self, filepath: str) -> None:
        """Save the database to a file (vectors and metadata)."""
        data = {
            "vectors": {key: vector.tolist() for key, vector in self.vectors.items()},
            "metadata": dict(self.metadata)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_from_file(self, filepath: str) -> None:
        """Load the database from a file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.vectors = defaultdict(np.array)
        self.metadata = defaultdict(dict)
        
        for key, vector_list in data["vectors"].items():
            self.vectors[key] = np.array(vector_list)
        
        for key, metadata in data["metadata"].items():
            self.metadata[key] = metadata


if __name__ == "__main__":
    # Example usage
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]
    
    # Example metadata
    metadata_list = [
        {"category": "food", "sentiment": "positive", "length": len(list_of_text[0])},
        {"category": "food", "sentiment": "neutral", "length": len(list_of_text[1])},
        {"category": "animals", "sentiment": "positive", "length": len(list_of_text[2])},
        {"category": "animals", "sentiment": "positive", "length": len(list_of_text[3])},
        {"category": "animals", "sentiment": "positive", "length": len(list_of_text[4])},
    ]

    async def test_enhanced_db():
        vector_db = EnhancedVectorDatabase()
        vector_db = await vector_db.abuild_from_list(list_of_text, metadata_list)
        
        print("=== Basic Search ===")
        results = vector_db.search_by_text("I think fruit is awesome!", k=2)
        for text, score, metadata in results:
            print(f"Text: {text}")
            print(f"Score: {score:.3f}")
            print(f"Metadata: {metadata}")
            print()
        
        print("=== Filtered Search (animals only) ===")
        filtered_results = vector_db.search_by_text(
            "cute pets", 
            k=3, 
            metadata_filter={"category": "animals"}
        )
        for text, score, metadata in filtered_results:
            print(f"Text: {text}")
            print(f"Score: {score:.3f}")
            print(f"Category: {metadata.get('category')}")
            print()
        
        print("=== Different Distance Metric (Euclidean) ===")
        euclidean_results = vector_db.search_by_text(
            "I love animals", 
            k=2, 
            distance_measure="euclidean"
        )
        for text, score, metadata in euclidean_results:
            print(f"Text: {text}")
            print(f"Euclidean Distance: {score:.3f}")
            print(f"Metadata: {metadata}")
            print()
        
        print("=== Database Statistics ===")
        stats = vector_db.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")

    # Run the test
    asyncio.run(test_enhanced_db())
