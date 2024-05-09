import os
from uuid import uuid4
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "text_embeddings"

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings_generator = OpenAIEmbeddings(api_key=openai_api_key)

def create_collection(client, collection_name):
    try:
        client.get_collection(collection_name)
        print(f"Collection {collection_name} already exists.")
    except Exception as e:
        print(f"Creating collection {collection_name} because it was not found: {e}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )
        print(f"Collection {collection_name} created.")

def chunk_data(text, chunk_size=500, chunk_overlap=10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def generate_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        try:
            embedding = embeddings_generator.embed_query(chunk)
            if isinstance(embedding, list) and len(embedding) == 1536:
                embeddings.append(embedding)
                print(f"Valid embedding received: {embedding[:10]}")
            else:
                print("Invalid embedding format:", embedding)
                embeddings.append(None)
        except Exception as e:
            print("Error generating embedding:", e)
            embeddings.append(None)
    return embeddings

def store_in_qdrant(text_chunks, embeddings):
    chunk_ids = []
    for text, embedding in zip(text_chunks, embeddings):
        if embedding is not None:
            chunk_id = str(uuid4())
            qdrant_client.upsert(
                collection_name=collection_name,
                points=[{
                    "id": chunk_id,
                    "vector": embedding,
                    "payload": {"text": text}
                }]
            )
            chunk_ids.append(chunk_id)
            print(f"Stored document with ID: {chunk_id}")
        else:
            print("Skipping storage due to missing embedding.")
    return chunk_ids

def process_text(input_text):
    chunks = chunk_data(input_text)
    embeddings = generate_embeddings(chunks)
    chunk_ids = store_in_qdrant(chunks, embeddings)
    return chunk_ids  # Return list of chunk IDs for later reference or deletion

def remove_text(client, collection_name, chunk_id):
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=[chunk_id])
        )
        print(f"Deleted chunk with ID: {chunk_id}")
    except Exception as e:
        print(f"Failed to delete chunk with ID: {chunk_id}. Error: {e}")

def cosine_search(client, collection_name, input_text, top_k=3):
    try:
        # Generate embedding for the input text
        embedding = embeddings_generator.embed_query(input_text)
        if not isinstance(embedding, list) or len(embedding) != 1536:
            print("Failed to generate a valid embedding.")
            return []

        # Perform search
        response = client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=top_k,
            search_params=models.SearchParams(
                hnsw_ef=128,
                exact=False
            )
        )

        # Since the response is a list of ScoredPoint objects directly
        if isinstance(response, list):
            results = [{'id': point.id, 'score': point.score, 'text': point.payload.get('text')} for point in response]
            return results
        else:
            print("Unexpected response structure:", response)
            return []
    except Exception as e:
        print(f"Error during search: {e}")
        return []

if __name__ == "__main__":
    example_text = "This is a string to search similar items for"
    chunk_ids = process_text(example_text)
    print("Chunk IDs:", chunk_ids)

    if chunk_ids:
        chunk_id_to_delete = chunk_ids[0]
        remove_text(qdrant_client, collection_name, chunk_id_to_delete)

    test_text = "What is a string?"
    search_results = cosine_search(qdrant_client, collection_name, test_text)
    print("Search Results:", search_results)
