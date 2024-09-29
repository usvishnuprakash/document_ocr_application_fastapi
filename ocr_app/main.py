from .packages.models import Base, PDFData, Conversation, User, TokenModel
from .packages.utils import *
from .packages.const import *
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
import io
import numpy as np
import cv2
import re
import fitz  # PyMuPDF
from pydantic import BaseModel
from transformers import pipeline
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]


# file imports


extracted_text = ""


# Set up the database connection
DATABASE_URL = "sqlite:///./chatbot.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the database tables
Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="OCR Application",
    description="An Api to extract text from image using ocr",
    version="1.0.0",
    docs_url="/docs",
    swagger_ui_parameters={
        "displayRequestDuration": True,

    },
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ! models
class QuestionModel(BaseModel):
    pdf_id: str
    question: str


class Token(BaseModel):
    access_token: str
    token_type: str


class EmailModel(BaseModel):
    email: str

# !


# Constants for JWT token generation
SECRET_KEY = "hello_happy_day"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    db = SessionLocal()
    user = db.query(User).filter(User.email == email).first()
    db.close()

    if user is None:
        raise credentials_exception
    return user


@app.get('/')
async def root():
    return {"message": "Welcome to the ocr api"}


@app.post("/login", response_model=Token)
async def login_for_access_token(email: EmailModel):
    db = SessionLocal()
    user = db.query(User).filter(User.email == email.email).first()
    if not user:
        user = User(email=email.email)
        db.add(user)
        db.commit()
        db.refresh(user)

    # create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.email,

        },
        expires_delta=access_token_expires)

    # token save
    token_entry = TokenModel(token=access_token, user_id=user.id)
    db.add(token_entry)
    db.commit()
    db.close()

    return {"access_token": access_token, "token_type": "bearer"}

# protected route


@app.get("/protected-route/")
async def protected_route(current_user: User = Depends(get_current_user)):
    try:
        return {"message": f"Hello {current_user.email}, you're authenticated!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.post('/extract-text/')
async def extract_text_from_image(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):

    try:
        global extracted_text
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', ".gif", 'tiff')):
            raise HTTPException(
                status_code=400, detail="Invalid file type. Please upload an image file")

        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        image_np = np.array(image)

        # correct skew

        image_np = correct_skew(image_np)

        # preprocessing
        preprocessed_image = preprocess_image(image_np)

        # convert to pil image
        pil_image = Image.fromarray(preprocessed_image)

        # pytesseract
        text = pytesseract.image_to_string(
            pil_image)

        text = clean_ocr_output(text)
        text = fix_line_breaks(text)

        # text = perform_ocr(pil_image)

        return JSONResponse(content={
            "extracted_text": text
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.post('/extract-document-text')
async def extract_document_text(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    try:
        # global extracted_text

        if not file.filename.lower().endswith(".pdf"):
            raise
        pdf_bytes = await file.read()
        extracted_text = extract_text_from_pdf(pdf_bytes)

        # save data to db
        db = SessionLocal()
        new_pdf = PDFData(extracted_text=extracted_text)
        db.add(new_pdf)
        db.commit()
        db.refresh(new_pdf)
        pdf_id = new_pdf.id
        db.close()
        #

        if not extracted_text:
            raise HTTPException(
                status_code=500, detail="Could not extract text from the PDF.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

    return {"message": "PDF uploaded successfully, and text extracted.", "extracted_text_preview": extracted_text, "pdf_id": pdf_id, }


def get_pdf_text(pdf_id: str) -> str:
    db = SessionLocal()
    pdf_data = db.query(PDFData).filter(PDFData.id == pdf_id).first()
    db.close()
    if not pdf_data:
        return None
    return pdf_data.extracted_text


def get_conversation_history(pdf_id: str):
    db = SessionLocal()
    conversations = db.query(Conversation).filter(
        Conversation.pdf_id == pdf_id).all()
    db.close()
    return conversations


def add_conversation(pdf_id: str, question: str, answer: str):
    db = SessionLocal()
    new_conversation = Conversation(
        pdf_id=pdf_id,
        question=question,
        answer=answer
    )
    db.add(new_conversation)
    db.commit()
    db.refresh(new_conversation)
    db.close()
    return new_conversation


@app.post("/ask-question")
async def ask_question(question: QuestionModel, current_user: User = Depends(get_current_user)):

    try:
        # global extracted_text
        extracted_text = get_pdf_text(question.pdf_id)

        if not extracted_text:
            raise HTTPException(
                status_code=400, detail="No PDF content found. Please upload a PDF first.")

        # retrieve conversation history
        conversation_history = get_conversation_history(question.pdf_id)

        # build context
        context = extracted_text

        for entry in conversation_history:
            context += f"Q:{entry.question}\nA:{entry.answer}\n"
        print("context", context)

        result = qa_pipeline(question=question.question,
                             context=context)

        # ADD conversation
        add_conversation(question.pdf_id, question.question, result["answer"])

        return {
            "question": question.question,
            "answer": result["answer"],
            "score": result["score"],
            "pdf_id": question.pdf_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.get('/conversation-history/{pdf_id}')
async def get_history(pdf_id: str, current_user: User = Depends(get_current_user)):
    try:
        conversation_history = get_conversation_history(pdf_id)
        if not conversation_history:
            raise HTTPException(
                status_code=404, detail="No conversation found for this PDF.")

        history = [{"question": conv.question, "answer": conv.answer, "timestamp": conv.timestamp}
                   for conv in conversation_history]
        return {"pdf_id": pdf_id, "history": history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


original_openapi = app.openapi
# Customize OpenAPI (Swagger) to use Bearer token


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    # Call the original openapi function to get the schema
    openapi_schema = original_openapi()

    # Define Bearer authentication in security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }

    # Apply Bearer authentication globally
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            openapi_schema["paths"][path][method]["security"] = [
                {"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Set custom OpenAPI definition
app.openapi = custom_openapi
