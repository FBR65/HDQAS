from qdrant_client import QdrantClient, models
import time
from tika import parser
from datetime import datetime as dt
import logging
import secrets
import uuid
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from openai import OpenAI
from py2neo import Graph, Node, Relationship
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
)

from __init__ import (
    TIKA_SERVER_URL,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_TIMEOUT,
    BASE_URL,
    API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    SENTENCE_TRANSFORMER_MODEL,
    SPACY_MODEL,
    NEO4J_URI,
    NEO4J_AUTH,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Document:
    """
    Is giving back Document(inhalt=content, metadaten=metadata)
    Needed to get a structured output for the parsed Document
    """

    def __init__(self, inhalt=None, metadaten=None):
        """
        Initialize a Document instance with content and metadata.
        Args:
            inhalt: The content of the document
            metadaten: The metadata of the document
        """
        self.inhalt = inhalt
        self.metadaten = metadaten


def semantic_chunking(text, threshold_percentile=25):
    """
    Splits text into semantic chunks based on sentence similarity.

    This function:
    1. Splits text into sentences
    2. Generates embeddings for each sentence
    3. Calculates cosine similarity between adjacent sentences
    4. Groups sentences into chunks based on similarity threshold

    Args:
        text (str): The text to be split into chunks
        threshold_percentile (int): Percentile value (0-100) to determine similarity
            threshold. Lower values create more chunks. Default: 25

    Returns:
        list: A list of text chunks where each chunk contains semantically
            related sentences
    """
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    embeddings = model.encode(sentences)
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        similarities.append(sim)
    threshold = np.percentile(similarities, threshold_percentile)
    chunks, current_chunk = [], []
    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        if i < len(similarities) and similarities[i] < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


class DataIntake:
    """
    A class to handle the ingestion and processing of documents into vector and graph databases.

    This class manages the entire document intake pipeline including:
    - Document parsing and text extraction
    - Semantic chunking of text
    - Vector embeddings generation
    - Storage in Qdrant vector database
    - Entity extraction and relationship mapping
    - Storage in Neo4j graph database

    Attributes:
        collection_name (str): Name of the Qdrant collection
        file_path (str): Path to the input document
        client (OpenAI): OpenAI client for embeddings
        embedding_model (str): Name of the embedding model
        qdrant_client (QdrantClient): Client for vector database operations
        local_model (SentenceTransformer): Fallback embedding model
        graph (Graph): Neo4j graph database connection
        chunk_hierarchy (dict): Tracks parent-child relationships between chunks
    """

    def __init__(self, collection_name, file_path) -> None:
        """
        Initialize the data_intake object with dense embedding model,
        a Qdrant client, and the Neo4j graph connection.
        """
        self.collection_name = collection_name
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
        )
        self.embedding_model = EMBEDDING_MODEL
        self.qdrant_client = QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT
        )
        self.file_path = file_path
        global model
        model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, trust_remote_code=True)
        global nlp
        nlp = spacy.load(SPACY_MODEL)

        # Initialize local embedding model as fallback
        try:
            self.local_model = SentenceTransformer("all-MiniLM-L6-v2")
            logging.info("Initialized local embedding model as fallback")
        except Exception as e:
            logging.error(f"Could not load local embedding model: {e}")
            self.local_model = None

        self.graph = Graph(NEO4J_URI, auth=NEO4J_AUTH)
        # Initialize NER pipeline
        self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")
        # Initialize Relation Extraction pipeline
        self.relation_extraction_model_name = "dslim/bert-base-NER"
        try:
            self.relation_extraction_tokenizer = AutoTokenizer.from_pretrained(
                self.relation_extraction_model_name
            )
            self.relation_extraction_model = (
                AutoModelForTokenClassification.from_pretrained(
                    self.relation_extraction_model_name
                )
            )
        except Exception as e:
            logging.warning(
                f"Could not load relation extraction model '{self.relation_extraction_model_name}': {e}"
            )
            self.relation_extraction_tokenizer = None
            self.relation_extraction_model = None

        # Add chunk tracking
        self.chunk_hierarchy = {}
        self.current_parent_id = None

    def organize_intake(self):
        """
        Orchestrates the complete document intake process.

        Steps:
        1. Streams and parses the document
        2. Creates necessary Qdrant collections
        3. Splits text into semantic chunks
        4. Processes chunks and stores in databases

        Returns:
            str: Completion timestamp
        """
        logging.info(f"Intake process started at: {dt.now()}")
        text = self.stream_document(self.file_path)
        logging.info(f"Stream Document ended at: {dt.now()}")
        logging.info(f"Create Qdrant Collection started at: {dt.now()}")
        create_collection = self.client_collection()
        logging.info(f"Create Qdrant Collection ended at: {dt.now()}")
        if create_collection:
            logging.info(f"Chunk Splitting started at: {dt.now()}")
            chunk_files = self.split_into_chunks(text.inhalt)
            logging.info(f"Chunk Splitting ended at: {dt.now()}")
            logging.info(f"DB Intake (Qdrant and Neo4j) started at: {dt.now()}")
            self.fill_database(chunk_files, text.metadaten)
            logging.info(f"DB Intake (Qdrant and Neo4j) ended at: {dt.now()}")
        logging.info(f"Intake process ended at: {dt.now()}")
        return f"Finished at: {dt.now()}"

    def stream_document(self, path):
        """
        Parses a document using Apache Tika and extracts content and metadata.

        Args:
            path (str): Path to the document file

        Returns:
            Document: Object containing document content and metadata
        """
        logging.info("Stream Document started.")
        parsed = parser.from_file(path, serverEndpoint=TIKA_SERVER_URL)
        if "resourceName" in parsed["metadata"]:
            if isinstance(parsed["metadata"]["resourceName"], list):
                decoded_text = parsed["metadata"]["resourceName"][0].strip("b'")
            else:
                decoded_text = parsed["metadata"]["resourceName"].strip("b'")
            parsed["metadata"]["file_name"] = decoded_text
            del parsed["metadata"]["resourceName"]
        content = parsed["content"]
        metadata = parsed["metadata"]
        document = Document(inhalt=content, metadaten=metadata)
        return document

    def generate_point_id(self):
        """
        Generates a unique identifier for vector database points using UUID4.

        Returns:
            str: Modified UUID string for use as point ID
        """
        uuid_value = uuid.uuid4().hex
        modified_uuid = "".join(
            (
                hex((int(c, 16) ^ secrets.randbits(4) & 15 >> int(c) // 4))[2:]
                if c in "018"
                else c
            )
            for c in uuid_value
        )
        logging.info(f"Created point id '{modified_uuid}'.")
        return str(modified_uuid)

    def client_collection(self):
        """
        Creates Qdrant collections with different distance metrics and payload indexing.

        Returns:
            str: Status of collection creation
        """
        collection_distances = ["COSINE"]
        for distances in collection_distances:
            collection = self.collection_name + "_" + str(distances)
            match distances:
                case "COSINE":
                    distance = models.Distance.COSINE
                case "EUCLID":
                    distance = models.Distance.EUCLID
                case "DOT":
                    distance = models.Distance.DOT
                case "MANHATTAN":
                    distance = models.Distance.MANHATTAN
            if not self.qdrant_client.collection_exists(
                collection_name=f"{collection}"
            ):
                self.qdrant_client.create_collection(
                    collection_name=collection,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=EMBEDDING_DIMENSION, distance=distance
                        )
                    },
                )
                logging.info(
                    f"Created collection '{collection}' in Qdrant vector database."
                )
                self.qdrant_client.create_payload_index(
                    collection_name=f"{collection}",
                    field_name="text",
                    field_schema=models.TextIndexParams(
                        type="text",
                        tokenizer=models.TokenizerType.WORD,
                        min_token_len=2,
                        max_token_len=15,
                        lowercase=True,
                    ),
                )
                logging.info(f"Created payload index for collection '{collection}'.")
        return "created"

    def split_into_chunks(self, text, output_dir="temp_chunks"):
        """
        Splits document text into semantic chunks using sentence embeddings.

        Args:
            text (str): Document text to split
            output_dir (str): Directory for temporary chunk storage

        Returns:
            list: Paths to generated chunk files
        """
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                print(f"Error creating directory '{output_dir}': {e}")

        chunk_files = []
        chunks = semantic_chunking(text)
        chunk_index = 0
        current_parent_id = str(uuid.uuid4())

        for i, chunk in enumerate(chunks):
            # Create new parent every 5 chunks
            if i % 5 == 0:
                current_parent_id = str(uuid.uuid4())

            chunk_filename = os.path.join(output_dir, f"chunk_{chunk_index}.json")
            chunk_id = str(uuid.uuid4())

            chunk_data = {
                "text": chunk,
                "metadata": {
                    "chunk_id": chunk_id,
                    "parent_id": current_parent_id,
                    "position": i,
                },
            }

            with open(chunk_filename, "w", encoding="utf-8") as f:
                json.dump(chunk_data, f)

            self.chunk_hierarchy[chunk_id] = current_parent_id
            chunk_files.append(chunk_filename)
            chunk_index += 1

        return chunk_files

    def extract_graph_data(self, chunk, metadata):
        """
        Extracts named entities and relationships from text using NLP models.

        Args:
            chunk (str): Text chunk to analyze
            metadata (dict): Document metadata

        Returns:
            tuple: (entities, relationships) found in the text
        """
        doc = nlp(chunk)
        resolved_text = self._resolve_coreferences(doc)
        resolved_doc = nlp(resolved_text)

        entities = [(ent.text, ent.label_) for ent in resolved_doc.ents]
        relationships = []

        # Identify entities using NER on the resolved text
        ner_results = self.ner_pipeline(resolved_text)
        entity_spans = []
        for entity in ner_results:
            entity_spans.append(
                (entity["start"], entity["end"], entity["entity"], entity["word"])
            )

        # Extract relationships between entities
        if (
            self.relation_extraction_model
            and self.relation_extraction_tokenizer
            and entity_spans
        ):
            for i in range(len(entity_spans)):
                for j in range(i + 1, len(entity_spans)):
                    head_span = entity_spans[i]
                    tail_span = entity_spans[j]
                    if "PERSON" in head_span[2] and "ORG" in tail_span[2]:
                        relationships.append((head_span[3], "WORKS_FOR", tail_span[3]))
                    elif "ORG" in head_span[2] and "LOC" in tail_span[2]:
                        relationships.append((head_span[3], "LOCATED_IN", tail_span[3]))

        return entities, relationships

    def _resolve_coreferences(self, doc):
        """
        Performs rule-based coreference resolution on spaCy document.

        Args:
            doc (spacy.Doc): Processed spaCy document

        Returns:
            str: Text with resolved coreferences
        """
        mentions = {}
        resolved_text = doc.text
        named_entities = {ent.text: ent.label_ for ent in doc.ents}

        for token in doc:
            if token.pos_ == "PRON":
                for ancestor in token.ancestors:
                    if (
                        ancestor.pos_ in ["NOUN", "PROPN"]
                        and ancestor.text in named_entities
                    ):
                        mentions[token.text] = ancestor.text
                        break

        for mention, referent in mentions.items():
            resolved_text = resolved_text.replace(mention, referent)

        return resolved_text

    def create_graph_nodes_and_relationships(
        self, entities, relationships, chunk_id, metadata
    ):
        """
        Creates nodes and relationships in Neo4j graph database.

        Args:
            entities (list): Extracted named entities
            relationships (list): Extracted relationships
            chunk_id (str): Unique identifier for the chunk
            metadata (dict): Document metadata
        """
        tx = self.graph.begin()

        # Create document chunk node with parent reference
        source_node = Node("DocumentChunk")
        source_node["id"] = chunk_id
        source_node["parent_id"] = self.chunk_hierarchy.get(chunk_id)
        source_node["source"] = metadata.get("file_name")
        tx.merge(source_node, "DocumentChunk", "id")

        # If this is a parent node, create parent-child relationship
        if chunk_id in self.chunk_hierarchy.values():
            parent_node = Node("ParentChunk")
            parent_node["id"] = chunk_id
            tx.merge(parent_node, "ParentChunk", "id")

        created_entities = {}
        for entity_text, entity_label in entities:
            if entity_text not in created_entities:
                # Create entity node
                entity_node = Node("Entity")
                entity_node["name"] = entity_text
                entity_node["label"] = entity_label
                # Merge on the "name" property
                tx.merge(entity_node, "Entity", "name")
                created_entities[entity_text] = entity_node

            # Create MENTIONS relationship
            mentions_rel = Relationship(
                source_node, "MENTIONS", created_entities[entity_text]
            )
            tx.merge(mentions_rel)

        # Create relationships between entities
        for head, relation, tail in relationships:
            head_node = created_entities.get(head)
            tail_node = created_entities.get(tail)
            if head_node and tail_node:
                try:
                    rel_type = relation.upper().replace(" ", "_")
                    rel = Relationship(head_node, rel_type, tail_node)
                    tx.merge(rel)
                except Exception as e:
                    logging.warning(
                        f"Could not create relationship '{relation}' between '{head}' and '{tail}': {e}"
                    )

        tx.commit()

    def fill_database(self, chunk_file_paths, metadata):
        """
        Processes chunks and stores them in vector and graph databases.

        Features:
        - Retries with exponential backoff
        - Fallback to local embedding model
        - Error handling and logging

        Args:
            chunk_file_paths (list): Paths to chunk files
            metadata (dict): Document metadata
        """
        collection_distances = ["COSINE"]
        max_retries = 5
        retry_delay = 3
        backoff_factor = 2

        for file_path in chunk_file_paths:
            points = []
            retry_count = 0
            current_delay = retry_delay
            use_local_model = False
            embedding_source = None  # Initialize embedding_source

            while retry_count < max_retries:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        chunk_data = json.load(f)
                        chunk = chunk_data["text"]

                    # Try OpenAI embeddings first, then fallback to local model
                    try:
                        if not use_local_model:
                            embedding_source = "openai"  # Set before API call
                            response = self.client.embeddings.create(
                                input=[chunk],
                                model=self.embedding_model,
                                timeout=30,
                            )
                            if not response or not response.data:
                                raise Exception("Empty response from OpenAI API")
                            dense_embedding = response.data[0].embedding
                        else:
                            embedding_source = "local"  # Set before local model call
                            dense_embedding = self.local_model.encode([chunk])[
                                0
                            ].tolist()

                    except Exception as emb_err:
                        logging.error(
                            f"Embedding error ({embedding_source}) for chunk file '{file_path}': {emb_err}"
                        )
                        if not use_local_model and self.local_model:
                            logging.info("Falling back to local embedding model...")
                            use_local_model = True
                            embedding_source = (
                                "local"  # Update source when falling back
                            )
                            continue
                        if retry_count < max_retries - 1:
                            retry_count += 1
                            time.sleep(current_delay)
                            current_delay *= backoff_factor
                            logging.info(
                                f"Retrying with delay of {current_delay} seconds..."
                            )
                            continue
                        raise

                    point_id = self.generate_point_id()
                    payload = {
                        "text": chunk,
                        "chunk_id": point_id,
                        "parent_id": self.chunk_hierarchy.get(point_id),
                        "embedding_source": embedding_source,
                    }

                    if dense_embedding is not None:
                        points.append(
                            models.PointStruct(
                                id=point_id,
                                vector={"dense": dense_embedding},
                                payload=payload,
                            )
                        )

                        # Upload to Qdrant with retry
                        if points:
                            for distances in collection_distances:
                                collection = self.collection_name + "_" + str(distances)
                                try:
                                    self.qdrant_client.upsert(
                                        collection_name=collection,
                                        points=points,
                                        wait=True,
                                    )
                                    logging.info(
                                        f"Successfully uploaded chunk from '{file_path}' to '{collection}' using {embedding_source} embeddings."
                                    )
                                except Exception as db_err:
                                    logging.error(f"Database upload error: {db_err}")
                                    if retry_count < max_retries - 1:
                                        retry_count += 1
                                        time.sleep(retry_delay)
                                        continue
                                    raise
                    else:
                        logging.warning(
                            f"Dense embedding was None for chunk: '{chunk[:50]}...'"
                        )

                    # Graph DB operations with retry
                    try:
                        entities, relationships = self.extract_graph_data(
                            chunk, metadata
                        )
                        self.create_graph_nodes_and_relationships(
                            entities, relationships, point_id, metadata
                        )
                    except Exception as graph_err:
                        logging.error(f"Graph DB error for '{file_path}': {graph_err}")
                        if retry_count < max_retries - 1:
                            retry_count += 1
                            time.sleep(retry_delay)
                            continue
                        raise

                    # If we get here, everything succeeded
                    break

                except Exception as e:
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        time.sleep(retry_delay)
                        logging.warning(
                            f"Attempt {retry_count} failed for '{file_path}': {e}. Retrying..."
                        )
                    else:
                        logging.error(
                            f"All attempts failed for '{file_path}' after {max_retries} retries: {e}"
                        )
                        break

            # Clean up the current chunk file
            try:
                os.remove(file_path)
            except Exception as e:
                logging.warning(f"Could not remove temp file '{file_path}': {e}")

        # Clean up temp directory if empty
        if os.path.exists("temp_chunks") and not os.listdir("temp_chunks"):
            try:
                os.rmdir("temp_chunks")
            except Exception as e:
                logging.warning(f"Could not remove temp directory: {e}")


if __name__ == "__main__":
    # Example usage of the DataIntake class
    file_path = "./data/41382-8.txt"
    collection_name = "sum_collection"
    intake = DataIntake(collection_name=collection_name, file_path=file_path)
    organized = intake.organize_intake()
    print(organized)
