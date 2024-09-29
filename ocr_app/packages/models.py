from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime


Base = declarative_base()


class PDFData(Base):
    __tablename__ = "pdf_data"
    id = Column(String, primary_key=True, index=True,
                default=lambda: str(uuid.uuid4()))
    extracted_text = Column(Text)


class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(String, primary_key=True, index=True,
                default=lambda: str(uuid.uuid4()))
    pdf_id = Column(String, ForeignKey("pdf_data.id"))
    question = Column(String)
    answer = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)


# User model
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True,
                default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)


# Token model
class TokenModel(Base):
    __tablename__ = "tokens"
    id = Column(String, primary_key=True, index=True,
                default=lambda: str(uuid.uuid4()))
    token = Column(String)
    user_id = Column(String, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")
