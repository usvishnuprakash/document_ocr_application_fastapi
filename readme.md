
```markdown
# OCR Chatbot Backend

## Technologies Used

- **FastAPI**: For building APIs
- **SQLAlchemy**: For database management
- **JWT**: For authentication
- **PyMuPDF**: For PDF handling
- **Transformers (Hugging Face)**: For handling the question-answering model

## Getting Started

### Prerequisites

- **Python 3.12** or higher is required.

### 1. Clone the Repository

```bash
git clone https://github.com/usvishnuprakash/document_ocr_application_fastapi.git
cd document_ocr_application_fastapi
```

### 2. Create a Virtual Environment (Python 3.12)

Ensure that Python 3.12 is installed, then create a virtual environment:

```bash
python3.12 -m venv venv
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

The app will now be running at `http://localhost:8000`.

### 5. API Documentation

Visit `http://127.0.0.1:8000/docs` for the automatically generated Swagger UI.

---

```

### Key Updates:
- **Python 3.12** is specified as a prerequisite.
- The virtual environment is created with Python 3.12 using `python3.12 -m venv venv` to ensure compatibility.
