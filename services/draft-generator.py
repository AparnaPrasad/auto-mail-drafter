import sqlite3
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from typing import List, Dict, Optional
from config import RULES
from email_service import EmailService  # Import from the same directory

# Import email-agent.py using importlib due to hyphen in filename
import importlib.util
spec = importlib.util.spec_from_file_location("email_agent", os.path.join(project_root, "agents", "email-agent.py"))
email_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(email_agent)
EmailRAGAgent = email_agent.EmailRAGAgent

def process_emails(db_path: str = "emails.db", rules: Dict[str, List[str]] = RULES) -> None:
    """
    Fetch emails from the database and process them through the EmailRAGAgent.
    
    Args:
        db_path: Path to the SQLite database
        rules: Dictionary of rules to apply for email processing
    """
    # Initialize the agent and email service
    agent = EmailRAGAgent()
    email_service = EmailService()
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Fetch all emails with message_id
        cursor.execute("SELECT message_id, sender, subject, body_text FROM emails") 
        emails = cursor.fetchall()
        
        # Process each email
        for email in emails:
            message_id, sender, subject, body_text = email  # Unpack the columns
            
            # Skip automated emails
            if "noreply" in sender.lower() or "alert" in sender.lower() or "aparnabs.prasad@gmail.com" not in sender:
                print(f"\nSkipping automated email from: {sender}")
                continue
                
            try:
                # Generate response
                response = agent.generate_response(body_text, rules or {})
                
                # Create draft reply
                original_message = {
                    'message_id': message_id,
                    'sender': sender,
                    'subject': subject,
                    'body': body_text
                }
                
                msg_id = email_service.create_draft_reply(original_message, response)
                
                if msg_id:
                    print(f"\nFrom: {sender}")
                    print(f"Original Email:\n{body_text}\n")
                    print(f"Generated Response:\n{response}\n")
                    print(f"Draft created with subject: Re: {subject}\n")
                    print("-" * 80)
                
            except Exception as e:
                print(f"Error processing email: {str(e)}")
                
    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    process_emails()
