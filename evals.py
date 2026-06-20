import os
from dotenv import load_dotenv
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate
from datasets import Dataset
from langchain_groq import ChatGroq
from app.rag_engine import rag_engine

load_dotenv()

# Initialize Groq LLM for RAGAS
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# Load document
print("Loading documents for evaluation...")
rag_engine.load_documents("./data/indian-penal-code.pdf", "test")

test_questions = [
    "What is section 420?",
    "What is punishment for theft?",
    "What is section 307?"
]

answers = []
contexts = []

print("Generating answers and retrieving contexts...")
for q in test_questions:
    # 1. Get the answer from your RAG engine
    answer, sources, confidence, ret_time, gen_time, cache_hit = rag_engine.query(q, "test")
    answers.append(answer)
    
    # 2. Fetch the actual text chunks to use as 'contexts' for RAGAS
    retriever = rag_engine.retrievers.get("test")
    if retriever:
        docs = retriever.invoke(q)[:5]  # Get top 5 chunks
        contexts.append([doc.page_content for doc in docs])
    else:
        contexts.append([""])

# Test dataset
test_data = {
    "question": [
        "What is section 420 of the Indian Penal Code?",
        "What is the punishment for theft under section 379?",
        "What does section 307 deal with?"
    ],
    "answer": answers,
    "contexts": contexts,
    "ground_truth": [
        "Section 420 deals with cheating and dishonestly inducing delivery of property",
        "Punishment for theft is imprisonment up to 3 years or fine or both",
        "Section 307 deals with attempt to murder"
    ]
}

dataset = Dataset.from_dict(test_data)

print("Running RAGAS evaluation...")
try:
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm
    )

    print("\n=== RAGAS Evaluation Results ===")
    print(results)
    print(f"\nFaithfulness: {results['faithfulness']:.3f}")
    print(f"Answer Relevancy: {results['answer_relevancy']:.3f}")
    print(f"Context Precision: {results['context_precision']:.3f}")
    print(f"Context Recall: {results['context_recall']:.3f}")
except Exception as e:
    print(f"Error during RAGAS evaluation: {str(e)}")