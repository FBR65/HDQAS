from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import NamedVector
from py2neo import Graph
import spacy
from sentence_transformers import SentenceTransformer

from __init__ import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_TIMEOUT,
    NEO4J_URI,
    NEO4J_AUTH,
    BASE_URL,
    API_KEY,
    EMBEDDING_MODEL,
    # EMBEDDING_DIMENSION,
    SPACY_MODEL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with its metadata"""

    chunk_id: str
    text: str
    score: float
    metadata: Dict
    source_type: str  # 'vector' or 'graph'
    entities: List[tuple] = None
    relationships: List[tuple] = None


class HybridRetriever:
    def __init__(self, collection_name: str):
        """Initialize retriever with necessary clients and models"""
        self.collection_name = collection_name

        # Initialize clients
        self.qdrant = QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT
        )

        # Try to get or create collection
        try:
            self.qdrant.get_collection(self.collection_name)
        except Exception as e:
            logger.warning(f"Collection not found: {e}")
            raise

        self.graph_db = Graph(NEO4J_URI, auth=NEO4J_AUTH)
        self.llm_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

        # Load models
        self.nlp = spacy.load(SPACY_MODEL)
        self.local_model = SentenceTransformer("all-MiniLM-L6-v2")

    async def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings"""
        try:
            # OpenAI client is synchronous, no need for await
            response = self.llm_client.embeddings.create(
                input=[text], model=EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting Embedding: {e}")
            raise

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.1,
        use_graph: bool = True,
        metadata_filter: Optional[Dict] = None,
    ) -> List[RetrievedChunk]:
        """
        Hybrid retrieval combining vector similarity and graph-based search
        """
        results = []

        # Get query embedding
        query_vector = await self.get_embedding(query)

        # Vector search in Qdrant using named vectors
        named_vector = NamedVector(name="dense", vector=query_vector)

        vector_results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=named_vector,
            limit=top_k,
            with_payload=True,
            score_threshold=min_score,
            query_filter=metadata_filter if metadata_filter else None,
        )
        print(f"Vector results: {vector_results}")
        # Process vector results from first query
        for point in vector_results:
            chunk = RetrievedChunk(
                chunk_id=point.id,
                text=point.payload["text"],
                score=point.score,
                metadata=point.payload.get("metadata", {}),
                source_type="vector",
            )
            results.append(chunk)

        # Graph-based retrieval if enabled
        if use_graph:
            # Extract entities from query
            query_doc = self.nlp(query)
            query_entities = [(ent.text, ent.label_) for ent in query_doc.ents]

            if query_entities:
                # Cypher query for graph traversal
                graph_query = """
                MATCH (d:DocumentChunk)-[:MENTIONS]->(e:Entity)
                WHERE e.name IN $entity_names
                WITH d, COUNT(DISTINCT e) as matches
                ORDER BY matches DESC
                LIMIT $limit
                RETURN d.id, d.source, matches
                """

                entity_names = [entity[0] for entity in query_entities]
                graph_results = self.graph_db.run(
                    graph_query, entity_names=entity_names, limit=top_k
                ).data()

                # Get full chunks for graph results
                for record in graph_results:
                    chunk_id = record["d.id"]

                    # Get chunk text from Qdrant
                    points = self.qdrant.retrieve(
                        collection_name=self.collection_name, ids=[chunk_id]
                    )

                    if points:
                        point = points[0]
                        # Get related entities and relationships
                        entities, relationships = self._get_graph_context(chunk_id)

                        chunk = RetrievedChunk(
                            chunk_id=chunk_id,
                            text=point.payload["text"],
                            score=record["matches"] / len(query_entities),
                            metadata=point.payload.get("metadata", {}),
                            source_type="graph",
                            entities=entities,
                            relationships=relationships,
                        )
                        results.append(chunk)

        # Sort combined results by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _get_graph_context(self, chunk_id: str) -> tuple[List[tuple], List[tuple]]:
        """Get entities and relationships for a chunk from Neo4j"""
        # Get entities
        entity_query = """
        MATCH (d:DocumentChunk {id: $chunk_id})-[:MENTIONS]->(e:Entity)
        RETURN e.name, e.label
        """
        entities = [
            (record["e.name"], record["e.label"])
            for record in self.graph_db.run(entity_query, chunk_id=chunk_id)
        ]

        # Get relationships
        rel_query = """
        MATCH (d:DocumentChunk {id: $chunk_id})-[:MENTIONS]->(e1:Entity)
        MATCH (e1)-[r]->(e2:Entity)
        WHERE type(r) in ['WORKS_FOR', 'LOCATED_IN']
        RETURN e1.name, type(r), e2.name
        """
        relationships = [
            (record["e1.name"], record["type(r)"], record["e2.name"])
            for record in self.graph_db.run(rel_query, chunk_id=chunk_id)
        ]

        return entities, relationships

    async def contextual_search(
        self,
        query: str,
        entity_type: Optional[str] = None,
        relationship_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Specialized search focusing on specific entity or relationship types
        """
        if entity_type:
            # Search for chunks with specific entity types
            graph_query = """
            MATCH (d:DocumentChunk)-[:MENTIONS]->(e:Entity)
            WHERE e.label = $entity_type
            RETURN DISTINCT d.id
            LIMIT 10
            """
            chunk_ids = [
                record["d.id"]
                for record in self.graph_db.run(graph_query, entity_type=entity_type)
            ]
        elif relationship_type:
            # Search for chunks with specific relationship types
            graph_query = """
            MATCH (d:DocumentChunk)-[:MENTIONS]->(e1:Entity)-[r]->(e2:Entity)
            WHERE type(r) = $rel_type
            RETURN DISTINCT d.id
            LIMIT 10
            """
            chunk_ids = [
                record["d.id"]
                for record in self.graph_db.run(graph_query, rel_type=relationship_type)
            ]
        else:
            return await self.retrieve(query)

        # Get full chunks and score them
        if chunk_ids:
            # query_vector = await self.get_embedding(query)
            results = []

            points = self.qdrant.retrieve(
                collection_name=self.collection_name,
                ids=chunk_ids,
                with_payload=True,
            )

            for point in points:
                chunk = RetrievedChunk(
                    chunk_id=point.id,
                    text=point.payload["text"],
                    score=0.0,  # Will be updated with semantic similarity
                    metadata=point.payload.get("metadata", {}),
                    source_type="contextual",
                )
                results.append(chunk)

            return results

        return []


# Example usage
async def main():
    # Initialize retriever
    retriever = HybridRetriever(collection_name="sum_collection_COSINE")

    # Basic hybrid search
    results = await retriever.retrieve(query="Admiral", top_k=5, use_graph=True)

    # Print results
    for result in results:
        print(f"\nChunk ID: {result.chunk_id}")
        print(f"Source: {result.source_type}")
        print(f"Score: {result.score:.3f}")
        print(f"Text: {result.text}")

        if result.entities:
            print("\nEntities:")
            for entity in result.entities:
                print(f"- {entity[0]} ({entity[1]})")

        if result.relationships:
            print("\nRelationships:")
            for rel in result.relationships:
                print(f"- {rel[0]} {rel[1]} {rel[2]}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
