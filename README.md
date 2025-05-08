# Hybrid Document Question-Answering System

A Python-based document analysis system that combines vector search and graph-based retrieval for intelligent document processing and question answering.

## Features

- **Hybrid Document Retrieval**: Combines vector similarity search with graph-based entity relationships
- **Smart Text Processing**: Semantic chunking and intelligent summarization
- **Multi-Database Integration**: Uses Qdrant for vector storage and Neo4j for graph relationships
- **Adaptive Summarization**: Map-reduce approach for handling large documents
- **Entity Recognition**: Built-in named entity recognition and relationship extraction
- **Error Handling**: Comprehensive error handling and fallback mechanisms
- **Progress Logging**: Detailed logging for monitoring and debugging

## Components

### 1. Data Intake (`intake.py`)
- Document parsing and text extraction
- Semantic chunking of text
- Vector embeddings generation
- Storage in Qdrant vector database
- Entity extraction and relationship mapping
- Storage in Neo4j graph database

### 2. Retriever (`retriever.py`)
- Hybrid search combining vector and graph approaches
- Entity-aware search capabilities
- Contextual and filtered searches
- Support for parent-child document relationships

### 3. Summarizer (`summarizer.py`)
- Automatic token counting and limit handling
- Direct and map-reduce summarization strategies
- Rate limiting and error handling
- Progress monitoring and logging

### 4. Main Interface (`main.py`)
- Unified question-answering interface
- Document summarization capabilities
- Flexible configuration options

## Dependencies

- OpenAI API for embeddings and completions
- Qdrant for vector storage
- Neo4j for graph database
- Sentence Transformers for local embeddings
- spaCy for NLP tasks
- Transformers for entity recognition
- Apache Tika for document parsing

## Environment Setup

Required environment variables:
- `TIKA_SERVER_URL`: URL for Apache Tika server
- `QDRANT_HOST`: Qdrant server host
- `QDRANT_PORT`: Qdrant server port
- `BASE_URL`: OpenAI API base URL
- `API_KEY`: OpenAI API key
- `NEO4J_URI`: Neo4j database URI
- `NEO4J_AUTH`: Neo4j authentication credentials

## Usage

1. **Document Ingestion**:
```python
intake = DataIntake(collection_name="your_collection", file_path="path/to/document")
intake.organize_intake()
```

2. **Question Answering**:
```python
answer = Answering(collection_name="your_collection")
result = await answer.answer(
    question="Your question?",
    use_type="retriever",
    max_tokens=4096,
    top_k=10,
    use_graph=True
)
```

3. **Document Summarization**:
```python
summarizer = QdrantSummarizer(collection_name="your_collection")
texts = summarizer.retrieve_all_texts()
summary = summarizer.summarize_texts(texts, max_tokens=4096)
```

## Error Handling

The system includes comprehensive error handling for:
- Rate limiting
- Context length exceeded
- API errors
- Database connection issues
- Token limit violations

## Performance Considerations

- Uses batching for large document processing
- Implements fallback mechanisms for embedding generation
- Supports both local and API-based models
- Includes retry logic for API calls

## License

The Code is Licensed under APGLv3. You may read the License in LICENSE.md