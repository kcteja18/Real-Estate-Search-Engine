"""
Real Estate Search Engine - ETL Pipeline
Phase 1: Data Ingestion & Storage

This script implements the complete ETL pipeline:
1. Read Excel file
2. Load AI model for floorplan parsing
3. Save canonical row data + AI model data into PostgreSQL  
4. Process certificates and extract PDF content
5. Prepare data for vector store indexing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Optional
import os
import re
import json
import sys
import torch
import torch.nn as nn
# --- AI Model Imports ---
import torch
try:
    from inference import RoomDetectionCNN, parse_floorplan
except ImportError:
    print("ERROR: Could not import 'floorplan_model.py'.")
    print("Please make sure it's in the same directory.")
    sys.exit(1)

# --- PDF processing ---
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# --- Vector database imports ---
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    SentenceTransformer = None

# --- Database imports ---
from database_schema import Property, Certificate, get_session, create_database, get_database_url
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class PropertyETL:
    """ETL Pipeline for Real Estate Property Data"""
    
    def __init__(self, excel_path: str = None, db_url: str = None):
        # File/DB Paths
        self.excel_path = excel_path or os.getenv('PROPERTY_DATA_FILE', r'D:\IITGN\Placement_prep\assets\assets\Property_list.xlsx')
        self.db_url = db_url or get_database_url('sqlite') # Use 'sqlite' or 'postgresql'
        self.assets_dir = Path(os.getenv('ASSETS_DIR', r'D:\IITGN\Placement_prep\assets\assets'))
        self.certificates_dir = self.assets_dir / 'certificates'
        self.image_dir = self.assets_dir / 'images'

        # AI Model Paths & Config
        self.model_checkpoint_path = os.getenv('MODEL_CHECKPOINT_PATH', r'D:\IITGN\Placement_prep\Real-Estate-Search-Engine\checkpoint_epoch_50.pth')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_rooms_per_type = 10 # Must match model training
        
        # Initialize database
        logger.info(f"Initializing database: {self.db_url}")
        self.engine = create_database(self.db_url)
        self.session = get_session(self.db_url)
        
        # Initialize AI Model
        self.model = self.load_ai_model()
        
        # Initialize vector database
        self.vector_db_path = Path(os.getenv('VECTOR_DB_PATH', './vector_db'))
        self.vector_db_path.mkdir(exist_ok=True)
        self.embedding_model = None
        self.faiss_index = None
        self.property_metadata = []
        
        # Statistics
        self.stats = {
            'total_properties': 0,
            'processed_properties': 0,
            'processed_certificates': 0,
            'vectorized_properties': 0,
            'errors': []
        }

    def load_ai_model(self) -> Optional[nn.Module]:
        """Load the floorplan parsing AI model"""
        logger.info(f"Loading AI model from {self.model_checkpoint_path}...")
        if not os.path.exists(self.model_checkpoint_path):
            logger.error(f"AI Model file not found at: {self.model_checkpoint_path}")
            return None
        try:
            model = RoomDetectionCNN(max_rooms_per_type=self.max_rooms_per_type)
            model.load_state_dict(torch.load(self.model_checkpoint_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            logger.info(f"✓ AI Model loaded. Using device: {self.device.upper()}")
            return model
        except Exception as e:
            logger.error(f"Error loading AI model: {e}")
            return None

    def extract_data(self) -> pd.DataFrame:
        """Step 1: Extract data from Excel file"""
        logger.info(f"Reading Excel file: {self.excel_path}")
        
        try:
            df = pd.read_excel(self.excel_path)
            logger.info(f"Successfully loaded {len(df)} properties")
            self.stats['total_properties'] = len(df)
            return df
        except Exception as e:
            error_msg = f"Error reading Excel file: {e}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize the data"""
        logger.info("Cleaning and normalizing data...")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Rename the problematic column name
        if 'title / short_description' in df.columns:
            df = df.rename(columns={'title / short_description': 'title'})
        
        # Extract City
        df['city'] = df.apply(
                        lambda row: (
                            match.group(1)
                            if (match := re.search(r'in\s+([A-Za-z]+)', str(row['title']))) 
                            and match.group(1).lower() in str(row['location']).lower()
                            else None
                        ),
                        axis=1
                    )
        df['location'] = df['location'].astype(str).str.replace('\n', ', ', regex=False)
        
        # Handle missing values
        df['certificates'] = df['certificates'].fillna('')
        df['seller_contact'] = df['seller_contact'].fillna(np.nan)
        df['metadata_tags'] = df['metadata_tags'].fillna('')
        df['image_file'] = df['image_file'].fillna('') # Ensure image_file col exists
        
        # Clean text fields
        text_fields = ['title', 'long_description', 'location', 'metadata_tags']
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].astype(str).str.strip()
        
        # Format phone numbers
        df['seller_contact'] = df['seller_contact'].apply(self.format_phone_number)
        
        # Ensure proper data types
        df['listing_date'] = pd.to_datetime(df['listing_date'])
        
        logger.info("Data cleaning completed")
        return df

    def format_phone_number(self, contact) -> Optional[str]:
        """Format phone numbers - convert 12-digit numbers starting with 91 to +91-XXXXXXXXXX format"""
        if pd.isna(contact):
            return None
        contact_str = str(contact).replace('.0', '').strip()
        digits_only = re.sub(r'[^\d]', '', contact_str)
        if len(digits_only) == 12 and digits_only.startswith('91'):
            return f"+91-{digits_only[2:]}"
        return contact_str if contact_str and contact_str != 'nan' else None

    def parse_certificates(self, certificates_str: str) -> List[str]:
        """Parse certificate string into list of filenames"""
        if not certificates_str or pd.isna(certificates_str):
            return []
        certificates = certificates_str.split('|')
        return [cert.strip() for cert in certificates if cert.strip()]

    def extract_pdf_text(self, pdf_path: Path) -> Optional[str]:
        """Extract text content from PDF file"""
        if not PyPDF2:
            logger.warning("PyPDF2 not available, skipping PDF text extraction")
            return None
        if not pdf_path.exists():
            logger.warning(f"PDF file not found: {pdf_path}")
            return None
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return None

    def process_certificates(self, property_id: str, certificates_str: str):
        """Process and store certificate information"""
        certificates = self.parse_certificates(certificates_str)
        for cert_filename in certificates:
            cert_path = self.certificates_dir / cert_filename
            existing = self.session.query(Certificate).filter_by(
                property_id=property_id, 
                filename=cert_filename
            ).first()
            if existing: continue
            
            extracted_text, file_size = None, None
            if cert_path.exists():
                extracted_text = self.extract_pdf_text(cert_path)
                file_size = cert_path.stat().st_size
            
            certificate = Certificate(
                property_id=property_id,
                filename=cert_filename,
                file_path=str(cert_path),
                extracted_text=extracted_text,
                file_size=file_size,
                is_processed=extracted_text is not None,
                processed_at=datetime.now() if extracted_text else None
            )
            self.session.add(certificate)
            self.stats['processed_certificates'] += 1

    def load_properties(self, df: pd.DataFrame):
        """Step 2 & 3: Run AI Model and Load property data into database"""
        logger.info("Loading properties into database...")
        
        for index, row in df.iterrows():
            try:
                # Check if property already exists
                existing = self.session.query(Property).filter_by(
                    property_id=row['property_id']
                ).first()
                if existing:
                    logger.info(f"Property {row['property_id']} already exists, skipping")
                    continue
                
                # --- AI Model Integration ---
                image_name = row.get('image_file')
                prediction_dict = {}
                if pd.notna(image_name) and self.model:
                    image_path = self.image_dir / image_name
                    prediction_dict = parse_floorplan(self.model, image_path, self.device)
                    if prediction_dict:
                        logger.info(f"  AI Success for {row['property_id']}: {json.dumps(prediction_dict)}")
                    else:
                        prediction_dict = {} # Ensure it's a dict
                else:
                    logger.warning(f"  Skipping AI model for {row['property_id']}: No image or model.")
                # --- End AI Model Integration ---

                # Create property record (mapping to database_schema.py)
                property_record = Property(
                    # Data from Excel
                    property_id=row['property_id'],
                    image_file = row['image_file'],
                    # num_rooms=int(row['num_rooms']),
                    # property_size_sqft=int(row['property_size_sqft']),
                    title=row['title'],
                    long_description=row['long_description'],
                    city = row['city'],
                    location=row['location'],
                    price=int(row['price']),
                    seller_type=row['seller_type'],
                    listing_date=row['listing_date'],
                    certificates=row['certificates'],
                    seller_contact=row['seller_contact'] if pd.notna(row['seller_contact']) else None,
                    metadata_tags=row['metadata_tags'],
                    
                    # AI Model Data (mapped to schema)
                    # 'rooms' (AI) -> bedrooms (DB)
                    # bedrooms=int(prediction_dict.get('rooms', 0)), 
                    # 'num_rooms' (Excel) -> rooms (DB) - for redundancy
                    rooms=int(row.get('rooms', 0)), 
                    halls=int(prediction_dict.get('halls', 0)),
                    kitchens=int(prediction_dict.get('kitchens', 0)),
                    bathrooms=int(prediction_dict.get('bathrooms', 0)),
                    garages=int(prediction_dict.get('garages', 0)),
                    
                    # Processing flags
                    is_indexed=False
                )
                
                self.session.add(property_record)
                
                # Process certificates for this property
                if row['certificates']:
                    self.process_certificates(row['property_id'], row['certificates'])
                
                self.stats['processed_properties'] += 1
                if (index + 1) % 10 == 0:
                    logger.info(f"Processed {index + 1} properties...")
                    
            except Exception as e:
                error_msg = f"Error processing property {row['property_id']}: {e}"
                logger.error(error_msg)
                self.stats['errors'].append(error_msg)
                continue

    def initialize_embedding_model(self):
        """Initialize the sentence transformer model for embeddings"""
        if not SentenceTransformer:
            logger.warning("sentence-transformers not available, skipping vector database creation")
            return False
        try:
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False

    def determine_approval_status(self, property_data: Dict) -> str:
        """Determine approval status based on certificates"""
        certificates = property_data.get('certificates', '')
        if not certificates: return "Not Approved"
        important_certs = ['green-building', 'fire-safety', 'structural-safety','pest-control']
        cert_count = sum(1 for cert in important_certs if cert in certificates.lower())
        if cert_count >= 3: return "Fully Approved"
        elif cert_count >= 1: return "Partially Approved"
        else: return "Not Approved"

    def create_property_text_for_embedding(self, property_data: Dict) -> str:
        """Create comprehensive text for embedding including approval status"""
        approval_status = self.determine_approval_status(property_data)
        
        # Create rich text representation
        text_parts = [
            f"Property ID: {property_data.get('property_id', '')}",
            f"Image: {property_data.get('image_file', '')}",
            f"Title: {property_data.get('title', '')}",
            f"Description: {property_data.get('long_description', '')}",
            f"Location: {property_data.get('location', '')}",
            f"City: {property_data.get('city', '')}",
            # f"Total Rooms (from Excel): {property_data.get('num_rooms', '')}",
            # f"Size: {property_data.get('property_size_sqft', '')} sqft",
            f"Price: ₹{property_data.get('price', '')}",
            f"Seller Type: {property_data.get('seller_type', '')}",
            f"Contact: {property_data.get('seller_contact', '')}",
            f"Tags: {property_data.get('metadata_tags', '')}",
            f"Approval Status: {approval_status}",
            f"Certificates: {property_data.get('certificates', '')}",
            # --- Add AI Data to Text ---
            f"AI Detected rooms: {property_data.get('rooms', 0)}",
            # f"AI Detected Bedrooms: {property_data.get('bedrooms', 0)}",
            f"AI Detected Bathrooms: {property_data.get('bathrooms', 0)}",
            f"AI Detected Kitchens: {property_data.get('kitchens', 0)}",
            f"AI Detected Halls: {property_data.get('halls', 0)}",
            f"AI Detected Garages: {property_data.get('garages', 0)}",
        ]
        
        return " | ".join(filter(None, text_parts))

    def create_vector_database(self):
        """Create FAISS vector database for semantic search"""
        if not faiss or not self.initialize_embedding_model():
            logger.warning("Vector database dependencies not available, skipping")
            return
            
        logger.info("Creating vector database...")
        
        try:
            # Get all properties from database
            properties = self.session.query(Property).all()
            if not properties:
                logger.warning("No properties found in database")
                return
            
            # Prepare texts for embedding
            texts, metadata = [], []
            
            for prop in properties:
                # This dict must match the DB schema
                property_data = {
                    'property_id': prop.property_id,
                    'image_file':prop.image_file,
                    'title': prop.title,
                    'long_description': prop.long_description,
                    'location': prop.location,
                    'city': prop.city,
                    # 'num_rooms': prop.num_rooms,
                    # 'property_size_sqft': prop.property_size_sqft,
                    'price': prop.price,
                    'seller_type': prop.seller_type,
                    'seller_contact': prop.seller_contact,
                    'metadata_tags': prop.metadata_tags,
                    'certificates': prop.certificates,
                    # --- Add AI Data to Metadata ---
                    # 'rooms': prop.rooms,
                    # 'bedrooms': prop.bedrooms,
                    'bathrooms': prop.bathrooms,
                    'kitchens': prop.kitchens,
                    'halls': prop.halls,
                    'garages': prop.garages,
                    'rooms': prop.rooms # The redundant one
                }
                
                text_for_embedding = self.create_property_text_for_embedding(property_data)
                texts.append(text_for_embedding)
                
                # Store metadata with approval status
                metadata_entry = property_data.copy()
                metadata_entry['approval_status'] = self.determine_approval_status(property_data)
                metadata.append(metadata_entry)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} properties...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            
            # Save FAISS index and metadata
            faiss_index_path = self.vector_db_path / "property_index.faiss"
            metadata_path = self.vector_db_path / "property_metadata.json"
            
            faiss.write_index(index, str(faiss_index_path))
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.stats['vectorized_properties'] = len(properties)
            logger.info(f"Vector database created successfully with {len(properties)} properties")
            logger.info(f"Index saved to: {faiss_index_path}")
            logger.info(f"Metadata saved to: {metadata_path}")
            
            # Update database to mark properties as indexed
            for prop in properties:
                prop.is_indexed = True
            
        except Exception as e:
            error_msg = f"Error creating vector database: {e}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)

    def run_etl(self):
        """Run the complete ETL pipeline"""
        logger.info("Starting ETL pipeline...")
        start_time = datetime.now()
        
        try:
            # Step 1: Extract
            df = self.extract_data()
            
            # Step 2: Clean
            df = self.clean_data(df)
            
            # Step 3: Load (Now includes AI model)
            self.load_properties(df)
            
            # Step 4: Create vector database
            self.create_vector_database()
            
            # Commit transaction
            self.session.commit()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Print statistics
            self.print_statistics(duration)
            logger.info("ETL pipeline completed successfully!")
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"ETL pipeline failed: {e}")
            raise
        finally:
            self.session.close()

    def print_statistics(self, duration: float):
        """Print ETL statistics"""
        print("\n" + "=" * 60)
        print("ETL PIPELINE STATISTICS")
        print("=" * 60)
        print(f"Total properties in Excel: {self.stats['total_properties']}")
        print(f"Properties processed: {self.stats['processed_properties']}")
        print(f"Certificates processed: {self.stats['processed_certificates']}")
        print(f"Properties vectorized: {self.stats['vectorized_properties']}")
        print(f"Processing duration: {duration:.2f} seconds")
        if duration > 0:
            print(f"Properties per second: {self.stats['processed_properties']/duration:.2f}")
        
        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:
                print(f"  - {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... and {len(self.stats['errors']) - 5} more errors")
        else:
            print("\n No errors encountered!")

def verify_data():
    """Verify the loaded data"""
    print("\n" + "=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)
    
    session = get_session(get_database_url('sqlite'))
    
    try:
        property_count = session.query(Property).count()
        print(f"Properties in database: {property_count}")
        certificate_count = session.query(Certificate).count()
        print(f"Certificates in database: {certificate_count}")
        
        # Show sample property with AI data
        sample = session.query(Property).filter(Property.bathrooms > 0).first()
        if not sample:
            sample = session.query(Property).first() # Get any if no AI data found
            
        if sample:
            print(f"\nSample Property:")
            print(f"  ID: {sample.property_id}")
            print(f"  Title: {sample.title}")
            print(f"  Excel Rooms: {sample.rooms}")
            # print(f"  AI Bedrooms: {sample.bedrooms}")
            print(f"  AI Bathrooms: {sample.bathrooms}")
            print(f"  AI Kitchens: {sample.kitchens}")
        
        # Vector database info
        vector_db_path = Path('./vector_db')
        faiss_index_path = vector_db_path / "property_index.faiss"
        if faiss_index_path.exists():
            print(f"\nVector Database: Ready for semantic search")
        else:
            print(f"\nVector Database: Not created")
        
    finally:
        session.close()

if __name__ == "__main__":
    # Run ETL pipeline
    etl = PropertyETL()
    etl.run_etl()
    
    # Verify results
    verify_data()