from helper_utils import project_embeddings, word_wrap, extract_text_from_pdf, load_chroma

from pypdf import PdfReader
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
import umap

load_dotenv()

# print(os.getenv("OPENAI_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PDF_FILE_PATH = "002_rag_advanced/data/microsoft-annual-report.pdf"
reader = PdfReader(PDF_FILE_PATH)
pdf_texts = [page.extract_text().strip() for page in reader.pages]
pdf_texts = [text for text in pdf_texts if text]
# pdf_texts = pdf_texts[:10]
print(f"Length of pdf texts: {len(pdf_texts)}")
# print(pdf_texts[0])
# print(pdf_texts[0] == "")
# print(pdf_texts[1])


from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

char_text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=0,
)
char_split_texts = char_text_splitter.split_text("\n\n".join(pdf_texts))
print(f"\nNumber of character split texts: {len(char_split_texts)}\n\n")
# print(char_split_texts[0])
# print("=========================chracter splitting texts ==================================\n\n")
# for text in char_split_texts:
#     print(text)
#     print("========================================\n\n")


token_text_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0,
    tokens_per_chunk=256
)
token_split_texts = []
for text in char_split_texts:
    token_split_texts += token_text_splitter.split_text(text)

print(f"\n\nNumber of token split texts: {len(token_split_texts)}\n\n")
# print("\n\n =========================token splitting texts ==================================\n\n")
# for text in token_split_texts:
#     print(text)
#     print("========================================\n\n")
# print(token_split_texts)


import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction()

ef = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.PersistentClient()
chroma_collection = chroma_client.get_or_create_collection(
    name="microsoft_collection",
    embedding_function=ef
)

ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.upsert(ids=ids, documents=token_split_texts)
print(chroma_collection.count())

# query = "What was the total revenue for the year ?"
# results = chroma_collection.query(query_texts=[query], n_results=5)
# retrieved_documents = results["documents"][0]
# print(retrieved_documents)


def augment_query_generated(query):
    prompt = """You are a helpful financial research assistant. Provide an example answer to the given 
    question, that might be found in a document like an annual report."""

    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": query
        }
    ]

    response = openai_client.chat.completions.create(
        messages=messages,
        model="gpt-5-nano"
    )
    answer = response.choices[0].message
    return answer


original_query = "What was the total profit for the year, and how does it compare to previous year ?"
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"{original_query} {hypothetical_answer}"

results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)

retrived_documents = results["documents"][0]


embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)


retrieved_embeddings = results["embeddings"][0]
original_query_embedding = ef([original_query])
augmented_query_embedding = ef([joint_query])

projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)
projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
)
projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)

import matplotlib.pyplot as plt

# Plot the projected query and retrieved documents in the embedding space
plt.figure()

plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot










