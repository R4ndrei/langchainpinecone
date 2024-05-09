import os
from uuid import uuid4
from dotenv import load_dotenv
from qdrant_client import QdrantClient
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

# Function to create the collection if it doesn't exist
def create_collection(client, collection_name):
    try:
        # Try to get collection information to check if it exists
        client.get_collection(collection_name)
        print(f"Collection {collection_name} already exists.")
    except Exception as e:
        # If the collection does not exist, create it
        print(f"Creating collection {collection_name} because it was not found: {e}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                'distance': 'Cosine',  # Adjust according to the distance metric relevant to your embeddings
                'size': 1536  # Dimensionality of your embeddings
            }
        )
        print(f"Collection {collection_name} created.")


# Ensure the collection exists before proceeding
create_collection(qdrant_client, collection_name)

# Function to split text into chunks
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

# Main function to process input text
def process_text(input_text):
    chunks = chunk_data(input_text)
    embeddings = generate_embeddings(chunks)
    chunk_ids = store_in_qdrant(chunks, embeddings)
    return chunk_ids  # Return list of chunk IDs for later reference or deletion

# Example usage
if __name__ == "__main__":
    example_text = "dgafgaf Test agdgggag"
    chunk_ids = process_text(example_text)
    print("Chunk IDs:", chunk_ids)
