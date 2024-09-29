
```markdown
# OCR Chatbot Backend

## Technologies Used

- **FastAPI**: For building APIs
- **SQLAlchemy**: For database management
- **JWT**: For authentication
- **PyMuPDF**: For PDF handling
- **Transformers (Hugging Face)**: For handling the question-answering model

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
uvicorn main:app --reload
```

The app will now be running at `http://127.0.0.1:8000`.

### 5. API Documentation

Visit `http://127.0.0.1:8000/docs` for the automatically generated Swagger UI.

---


```
