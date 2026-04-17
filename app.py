from src.rag.embeddings import get_embeddings
from src.rag.chunking import chunk_text
from src.rag.vector_store import create_faiss_index
from src.rag.retriever import retrieve
from src.rag.pdf_loader import load_pdfs
from src.rag.assistant import build_prompt, create_agent


def get_pdf_paths():
    paths = input("Enter PDF file paths separated by commas: ").strip()
    return [p.strip() for p in paths.split(",") if p.strip()]


pdf_paths = get_pdf_paths()
if not pdf_paths:
    raise SystemExit("No PDF files provided. Exiting.")

print(f"Loading PDFs: {pdf_paths}")
text = load_pdfs(pdf_paths)

# 🔹 STEP 2: Chunking (bigger chunks for better context)
chunks = chunk_text(text, size=500)

# 🔹 STEP 3: Embeddings
embeddings = get_embeddings(chunks)

# 🔹 STEP 4: Create FAISS index
index = create_faiss_index(embeddings)

print("Processing...")

# 🔹 STEP 5: Create Agent (better model)
agent = create_agent()

print("\nAI is ready! Ask your questions.\n")

# 🔹 STEP 6: Chat loop
while True:
    q = input("Ask: ")

    if q.lower() in ["exit", "quit"]:
        break

    # 🔥 STEP 7: Smart Retrieval
    if "summary" in q.lower():
        # 👉 For summary → retrieve more relevant chunks across all PDFs
        results = retrieve(q, chunks, index, k=min(8, len(chunks)))
        context = "\n".join(results)
    else:
        # 👉 For normal questions → top 3 chunks
        results = retrieve(q, chunks, index, k=3)
        context = "\n".join(results)

    # 🔥 STEP 8: Strong Prompt
    full_prompt = build_prompt(context, q, summary="summary" in q.lower())

    # 🔥 STEP 9: Get response
    response = agent.run(full_prompt)

    # 🔥 STEP 10: Clean output (IMPORTANT)
    final_answer = response.content.strip().split("\n")[0]

    print("\nAnswer:", final_answer, "\n")