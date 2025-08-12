from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader
from dotenv import load_dotenv
load_dotenv()

from helper_utils import project_embeddings, load_chroma, extract_text_from_pdf


pdf_file_path = "002_rag_advanced/data/microsoft-annual-report.pdf"
reader = PdfReader(pdf_file_path)
texts = [page.extract_text().strip() for page in reader.pages]
pdf_texts = [text for text in texts if text]
pdf_texts = pdf_texts[:10]


from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    separators=["\n\n", "\n", " ", ""],
    chunk_overlap = 0
)
char_split_texts = char_splitter.split_text("\n\n".join(pdf_texts))


token_text_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, 
    tokens_per_chunk=256
)
token_split_texts = []
for text in char_split_texts:
    token_split_texts += token_text_splitter.split_text(text)


import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ef = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="micorsoft2", embedding_function=ef
)

ids = [str(i) for i in range(len(token_split_texts))]
collection.upsert(
    ids=ids, documents=token_split_texts
)

from openai import OpenAI
import os
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
prompt = """You are a knowledgeable financial research assistant. Your users are inquiring about an annual
report. For the given question, propose up to five related questions to assist them in finding the information
they need. Provide concise, single-topic questions (without compounding sentences) that cover various aspects
of the topic. Ensure each question is complete and directly related to the original inquiry. 
List each question on a separate line without numbering."""
original_query = "What details can you provide about the factors that led to revenue growth ?"

messages  = [
    {
        "role": "system", 
        "content": prompt
    },
    {
        "role": "user",
        "content": original_query
    }
]

response = openai_client.chat.completions.create(
    messages=messages,
    model="gpt-5-nano"
)

print("\nOpenai response: \n\n")
print(response)
new_questions = response.choices[0].message.content.split("\n")

joint_query = [original_query] + new_questions


results = collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"]

# Deduplicate the retrieved documents
# for each query we will have different documents, so we need to deduplicate it
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)


# output the results documents
for i, documents in enumerate(retrieved_documents):
    print(f"Query: {joint_query[i]}")
    print("")
    print("Results:")
    for doc in documents:
        print(doc)
        print("\n\n\n")
    print("-" * 100)



import umap
embeddings = collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# 4. We can also visualize the results in the embedding space
original_query_embedding = ef([original_query])
augmented_query_embeddings = ef(joint_query)


project_original_query = project_embeddings(original_query_embedding, umap_transform)
project_augmented_queries = project_embeddings(
    augmented_query_embeddings, umap_transform
)

retrieved_embeddings = results["embeddings"]
result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]

projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)

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
    project_augmented_queries[:, 0],
    project_augmented_queries[:, 1],
    s=150,
    marker="X",
    color="orange",
)
plt.scatter(
    projected_result_embeddings[:, 0],
    projected_result_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    project_original_query[:, 0],
    project_original_query[:, 1],
    s=150,
    marker="X",
    color="r",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot