import sqlite3
import logging
import argparse
import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('services/email_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('services/email_service')

def clear_database():
    """Clear all content from the database tables."""
    try:
        # Get database path from config
        db_path = DATABASE_CONFIG["path"]
        
        print(f"Connecting to database at {db_path}...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Clear attachments table first (due to foreign key constraint)
        print("Clearing attachments table...")
        cursor.execute("DELETE FROM attachments")

        # Clear emails table
        print("Clearing emails table...")
        cursor.execute("DELETE FROM emails")
        
        # Drop emails table
        # print("Dropping emails table...")
        # cursor.execute("DROP TABLE IF EXISTS emails")
        
        # Reset auto-increment counters
        print("Resetting auto-increment counters...")
        cursor.execute("DELETE FROM sqlite_sequence")
        
        conn.commit()
        conn.close()
        print("Database tables cleared successfully")
        logger.info("Database tables cleared successfully")
    except Exception as e:
        error_msg = f"Error clearing database: {str(e)}"
        print(error_msg)
        logger.error(error_msg)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clear email database tables')
    
    try:
        clear_database()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.error(f"Fatal error: {e}")
