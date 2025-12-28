# RAG-based AI Teaching Assistant 🤖📘

A Retrieval-Augmented Generation (RAG) based AI system that enables you to build a **custom AI teaching assistant** using your own video-based educational content.  
The system retrieves relevant context from your data and generates accurate, context-aware answers using a Large Language Model (LLM).

---

## 🚀 How to Use This RAG AI Teaching Assistant on Your Own Data

### ✅ Step 1 — Collect Your Videos
Move all your lecture or educational video files into the `videos/` folder.

---

### ✅ Step 2 — Convert Videos to MP3
Convert all video files into `.mp3` format by running:

    python video_to_mp3.py

This step extracts audio from each video file.

---

### ✅ Step 3 — Convert MP3 to JSON
Convert all `.mp3` files into structured JSON transcripts by running:

    python mp3_to_json.py

Each JSON file contains cleaned and chunked text extracted from the audio.

---

### ✅ Step 4 — Convert JSON Files to Vectors
Generate semantic embeddings from the JSON transcripts by running:

    python preprocess_json.py

This script:
- Loads JSON transcript files  
- Generates text embeddings  
- Converts them into a dataframe  
- Saves the embeddings as a **joblib pickle file** for efficient retrieval  

---

### ✅ Step 5 — Prompt Generation and LLM Inference
Run the query pipeline using:

    python process_incoming.py

This step:
- Loads the joblib embedding file into memory  
- Retrieves the most relevant text chunks based on the user query  
- Creates a context-aware prompt  
- Feeds the prompt to the **LLM** to generate an accurate response  

---

## 🧠 How It Works (RAG Pipeline)

1. Data Ingestion  
   Videos → Audio (MP3) → JSON transcripts  

2. Embedding Generation  
   Text chunks → Vector embeddings  

3. Retrieval  
   Semantic similarity search over embeddings  

4. Generation  
   LLM generates grounded answers using retrieved context  

This approach significantly reduces hallucinations and improves answer relevance.

---

## 🛠️ Tech Stack

- Python  
- Natural Language Processing (NLP)  
- Text Embeddings  
- Vector Similarity Search  
- Joblib (vector storage)  
- Large Language Models (LLMs)  

---

## 📌 Use Cases

- AI Teaching Assistants  
- Course Question–Answering Systems  
- Internal Knowledge Base Chatbots  
- Lecture and Training Content Search  

---

👨‍💻 Author

Anshul Rajpoot
ECE Undergraduate | Data Science & ML Enthusiast
🔗 GitHub: https://github.com/Anshul-Rajpoot
