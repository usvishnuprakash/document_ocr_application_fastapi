
from .const import *
from jose import jwt, JWTError
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2AuthorizationCodeBearer
from PIL import Image
import pytesseract
import numpy as np
import cv2
import re
import fitz  # PyMuPDF
from transformers import pipeline
from typing import Optional
from datetime import datetime, timedelta


qa_pipeline = pipeline("question-answering",
                       model="bert-large-uncased-whole-word-masking-finetuned-squad")


def preprocess_image(image: np.ndarray) -> np.ndarray:
    # grey scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gaussianBlur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize the image using Otsu's thresholding

    _, binary_image = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # MEDIAN BLURRING
    denoised = cv2.medianBlur(binary_image, 3)

    height, width = denoised.shape
    scaling_factor = 2

    resized_image = cv2.resize(
        denoised, (scaling_factor*width, scaling_factor*height))
    return resized_image


# preprocess function for better OCR results
def advance_preprocess_image(image: np.ndarray) -> np.ndarray:
    # gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for better text detection under varying lighting conditions

    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply morphological operations to enhance the text
    kernal = np.ones((1, 1), np.uint8)
    morphed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernal)

    # Resize the image if necessary to imporve OCR accuracy
    height, width = morphed.shape
    scaling_factor = 2
    resized_image = cv2.resize(
        morphed, (scaling_factor * width, scaling_factor*height))

    return resized_image

#


def correct_skew(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold the image
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # find counters
    coords = np.column_stack(np.where(binary > 10))
    angle = cv2.minAreaRect(coords)[-1]

    # the angle needs to be adjusted
    if angle < -45:
        angle = -(90+angle)
    else:
        angle = -angle

    # rotate the image and covert the skew
    (h, w) = image.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def perform_ocr(image: Image.Image) -> str:
    # tesseract configuration
    custom_config = r'--oem 3 --psm 6'
    str_value = pytesseract.image_to_string(image, config=custom_config)
    return str_value


def clean_ocr_output(text: str) -> str:
    # Replace common misinterpretations by Tesseract
    text = text.replace('1eaf', 'leaf')   # Correct common word errors
    text = text.replace('De1ivery', 'Delivery')
    text = text.replace('Benga1uru', 'Bengaluru')

    # Common letter misinterpretations
    text = text.replace('O', '0')   # Replace "O" with "0" if needed
    # Correct "0" to "O" in cases like email/URLs
    text = text.replace('0', 'O')
    text = text.replace('l', '1')   # Replace lowercase "l" with "1"
    # Replace "1" back to "l" if mistakenly used
    text = text.replace('1', 'l')

    # Correct email addresses and URLs
    text = re.sub(r'\s+@\s+', '@', text)  # Fix spaces in email addresses
    text = re.sub(r'\s+com\b', '.com', text)  # Fix ". com" to ".com"
    text = re.sub(r'gmai1\.com', 'gmail.com', text)  # Common mistake in emails

    return text


def fix_line_breaks(text: str) -> str:
    # Replace excessive newlines with a single space
    # Replace single line breaks with spaces
    return re.sub(r'(?<!\n)\n(?!\n)', ' ', text)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted_text = ""

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        extracted_text += page.get_text("text") + "\n"
    return extracted_text


# !


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow()+expires_delta
    else:
        expire = datetime.utcnow()+timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})

    encode_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encode_jwt



