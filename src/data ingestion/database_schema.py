"""
Real Estate Search Engine - Database Schema
PostgreSQL schema design based on property data analysis
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

class Property(Base):
    """Main property table - structured data"""
    __tablename__ = 'properties'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    property_id = Column(String(20), unique=True, nullable=False, index=True)
    image_file = Column(String(200), nullable=False)
    title = Column(String(200), nullable=False)
    long_description = Column(Text, nullable=False)
    city = Column(String(100), nullable=True)  # Extracted from title/location
    location = Column(String(200), nullable=False)
    price = Column(Integer, nullable=False)
    seller_type = Column(String(20), nullable=False)
    listing_date = Column(DateTime, nullable=False)
    certificates = Column(String(500))  # Comma-separated certificate files
    seller_contact = Column(String(20))  # Formatted phone number
    metadata_tags = Column(String(300))
    rooms = Column(Integer,nullable=False)
    halls = Column(Integer,nullable=False)# JSON string of room details detected
    kitchens = Column(Integer,nullable=False)
    bathrooms = Column(Integer,nullable=False)
    garages = Column(Integer,nullable=False)
    # Additional fields for processing
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_indexed = Column(Boolean, default=False)  # Track if indexed in vector store
    
    def __repr__(self):
        return f"<Property(id='{self.property_id}', title='{self.title[:50]}...')>"

class Certificate(Base):
    """Certificate documents table"""
    __tablename__ = 'certificates'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    property_id = Column(String(20), nullable=False, index=True)
    filename = Column(String(100), nullable=False)
    file_path = Column(String(500), nullable=False)
    extracted_text = Column(Text)  # Extracted PDF text content
    file_size = Column(Integer)
    processed_at = Column(DateTime)
    is_processed = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<Certificate(property_id='{self.property_id}', filename='{self.filename}')>"

class SearchLog(Base):
    """Log user searches for analytics and improvement"""
    __tablename__ = 'search_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    search_type = Column(String(50))  # 'semantic', 'structured', 'hybrid'
    results_count = Column(Integer)
    user_session = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)
    response_time_ms = Column(Float)

def create_database(database_url: str):
    """Create database tables"""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return engine

def get_session(database_url: str):
    """Get database session"""
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    return Session()

# Database configuration
DATABASE_CONFIG = {
    'postgresql': {
        'driver': 'postgresql+psycopg2',
        'default_url': 'postgresql://realestate:password@localhost:5432/realestate_db'
    },
    'sqlite': {  # Fallback for development
        'driver': 'sqlite',
        'default_url': 'sqlite:///./realestate.db'
    }
}

def get_database_url(db_type='sqlite'):
    """Get database URL from environment or use default"""
    if db_type == 'postgresql':
        return os.getenv('DATABASE_URL', DATABASE_CONFIG['postgresql']['default_url'])
    else:
        return os.getenv('DATABASE_URL', DATABASE_CONFIG['sqlite']['default_url'])

if __name__ == "__main__":
    # Test database creation
    print("Creating database schema...")
    
    # Use SQLite for development
    db_url = get_database_url('sqlite')
    print(f"Database URL: {db_url}")
    
    engine = create_database(db_url)
    print("Database schema created successfully!")
    
    # Test connection
    session = get_session(db_url)
    print(f" Database connection test successful!")
    session.close()