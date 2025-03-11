%%writefile flask_compliance_ai.py
from flask import Flask, request, render_template
from pyngrok import ngrok
import openai
import json
import os
import fitz  # PyMuPDF for PDF extraction
import faiss
import numpy as np

# ✅ Set API Keys
openai.api_key = "your_openai_api_key_here"  # Replace with your real OpenAI key
ngrok.set_auth_token("your_ngrok_auth_token_here")  # Replace with your Ngrok token

# ✅ Flask app setup
app = Flask(__name__, template_folder="templates")

# ✅ Ensure templates exist
if not os.path.exists("templates"):
    os.makedirs("templates")

if not os.path.exists("templates/index.html"):
    with open("templates/index.html", "w") as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Compliance AI Assistant</title>
</head>
<body>
    <h2>Compliance AI Assistant</h2>
    <form method="POST">
        <label for="query">Enter your compliance question:</label>
        <input type="text" id="query" name="query" required>
        <button type="submit">Ask</button>
    </form>
    
    {% if response %}
        <h3>Response:</h3>
        <p>{{ response }}</p>
    {% endif %}
</body>
</html>""")

# ✅ Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        if text.strip():
            sections.append({"page": page_num + 1, "text": text.strip()})

    return sections

# ✅ Generate OpenAI vector embeddings
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# ✅ Create FAISS vector database
def create_faiss_index(pdf_texts):
    embeddings = [get_embedding(section["text"]) for section in pdf_texts]
    dimension = len(embeddings[0])

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))

    return index, pdf_texts

# ✅ Hybrid search (Keyword + Vector)
def keyword_search(query, pdf_texts):
    query_words = query.lower().split()
    results = [section for section in pdf_texts if any(word in section["text"].lower() for word in query_words)]
    return results[:3]

def vector_search(query, faiss_index, pdf_texts):
    query_embedding = np.array([get_embedding(query)], dtype=np.float32)
    _, indices = faiss_index.search(query_embedding, 3)
    return [pdf_texts[i] for i in indices[0]]

def hybrid_search(query, faiss_index, pdf_texts):
    keyword_results = keyword_search(query, pdf_texts)
    return keyword_results if keyword_results else vector_search(query, faiss_index, pdf_texts)

# ✅ AI Query Function
def ask_openai(query):
    relevant_sections = hybrid_search(query, faiss_index, pdf_texts)

    if not relevant_sections:
        return "No relevant compliance data found."

    context = "\n".join([
        f"- {section['text'][:200]}... (Page {section['page']})"
        for section in relevant_sections
    ])

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a compliance expert providing regulatory guidance."},
            {"role": "user", "content": f"Query: {query}\n\nRelevant Regulations:\n{context}\n\nProvide a clear and concise answer."}
        ]
    )

    return response.choices[0].message.content

# ✅ Flask Route
@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_query = request.form.get("query")
        response = ask_openai(user_query)
    
    return render_template("index.html", response=response)

# ✅ Load PDFs & Start System
pdf_texts = extract_text_from_pdf("your_compliance_document.pdf")  # Upload your PDF
faiss_index, pdf_texts = create_faiss_index(pdf_texts)

ngrok.kill()
public_url = ngrok.connect(5000).public_url
print(f"Public URL: {public_url}")

if __name__ == "__main__":
    app.run(port=5000)
