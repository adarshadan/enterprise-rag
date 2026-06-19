import os
from sentence_transformers import SentenceTransformer

# Define a local directory inside the container to save the model
MODEL_DIR = "/app/models/all-mpnet-base-v2"

print("Downloading and saving model locally...")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model.save(MODEL_DIR)
print(f"Model successfully saved to {MODEL_DIR}")