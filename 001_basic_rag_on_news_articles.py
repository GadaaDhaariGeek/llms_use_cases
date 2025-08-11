from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os
import sys


# functions to load text documents from a directory 
def load_text_documents(data_dir_path):
    documents = []
    for filename in os.listdir(data_dir_path):
        if filename.endswith(".txt"):
            with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as fr:
                documents.append({"id": filename, "text": fr.read().strip()})
    print(f"\nRead {len(documents)} documents.")
    return documents

# Simple function to divide the given doc into chunks
# implements a version of charactertextsplitting of langchain
def character_split_text(text, chunk_size, chunk_overlap):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i = i+chunk_size-chunk_overlap
    return chunks

def create_chunks_from_documents(documents, chunk_size, chunk_overlap):
    # print(f"\nChunking {len(documents)} documents.")
    chunked_documents = []
    for doc in documents:
        chunks = character_split_text(doc["text"], chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunked_documents.append({"id": f"{doc['id']}_chunk_{i+1}", "text": chunk})

    print(f"\nCreated {len(chunked_documents)} chunks from all documents.")
    return chunked_documents

def get_openai_embeddings(chunked_documents):
    for chunk in chunked_documents:
        response = openai_client.embeddings.create(input=chunk["text"], model=EMBEDDING_MODEL_NAME, dimensions=786)
        chunk["embedding"] = response.data[0].embedding
    return chunked_documents

def inserting_data_into_vector_store(chunked_documents):
    for chunk in chunked_documents:
        collection.upsert(ids=[chunk["id"]], documents=[chunk["text"]], embeddings=[chunk["embedding"]])

def load_and_insert_the_data_into_vector_store(data_dir_path, chunk_size, chunk_overlap):
    text_documents = load_text_documents(data_dir_path)
    chunked_documents = create_chunks_from_documents(text_documents, chunk_size, chunk_overlap)
    chunked_documents = get_openai_embeddings(chunked_documents)
    inserting_data_into_vector_store(chunked_documents)
    print("\n\nVector Store populated.")

def query_the_vector_store(question, n_results=2):
    results = collection.query(query_texts=question, n_results=n_results)
    relevant_chunks = [chunk for sublist in results["documents"] for chunk in sublist]
    return relevant_chunks

def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you do not know the "
        "answer then say that you do not know the answer. Use two sentences maximum to keep the answer concise. "
        "\n\nContext: \n" + context + "\n\nQuestion: \n" + question
    )


    response = openai_client.chat.completions.create(
        model = LLM_MODEL_NAME, 
        messages = [
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    answer = response.choices[0].message
    return answer

def runner(question):
    relevant_chunks = query_the_vector_store(question)
    answer = generate_response(question, relevant_chunks)
    return answer


if __name__ == "__main__":

    # load the environment
    load_dotenv()

    # setting up some variables
    DATA_DIR = "news_articles_data"
    EMBEDDING_MODEL_NAME = "text-embedding-3-small"
    LLM_MODEL_NAME = "gpt-5-nano"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CHROMADB_PATH = "persistent_storage_chromadb"
    COLLECTION_NAME = "news_articles_collection"
    chunk_size = 1000
    chunk_overlap = 50


    # creating openai client and embedding function
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBEDDING_MODEL_NAME)

    # Initializing chroma client 
    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)

    try:
        # Attempt to get the collection
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except chromadb.errors.NotFoundError:
        print(f"Collection '{COLLECTION_NAME}' does not exist. Creating it now...")
        # Create the collection here
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef
        )
        print("\n\n Loading the data into vector store.")
        load_and_insert_the_data_into_vector_store(DATA_DIR, chunk_size, chunk_overlap)

    
    while True:
        print("\nEnter a question you would like to ask your assistant. Press q to quit\n")
        question = input()
        if question.strip().lower() == "q":
            print("Exiting. Ok Bye.")
            sys.exit(0)
        answer = runner(question)
        print(answer.content)










