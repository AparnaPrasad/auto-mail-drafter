import imaplib
import email
import sqlite3
import datetime
import logging
from email.header import decode_header
from datetime import timedelta
import re
import sys
import os
from email.mime.text import MIMEText
from email.utils import formatdate
import time
from typing import Dict
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMAIL_CONFIG, DATABASE_CONFIG, FETCH_CONFIG

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

class EmailService:
    def _remove_html_tags(self, text):
        # Extract content between body tags if they exist
        body_match = re.search(r'<body[^>]*>(.*?)</body>', text, re.DOTALL | re.IGNORECASE)
        if body_match:
            text = body_match.group(1)
        
        # Remove all HTML tags from the extracted content
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', text)
        
        # Remove multiple blank lines (more than 2 consecutive)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()

    def __init__(self):
        """Initialize the email service with configuration."""
        try:
            # Load configuration from config.py
            self.email_address = EMAIL_CONFIG["address"]
            self.password = EMAIL_CONFIG["password"]
            self.imap_server = EMAIL_CONFIG["imap_server"]
            self.imap_port = EMAIL_CONFIG["imap_port"]
            self.db_path = DATABASE_CONFIG["path"]
            self.use_ssl = EMAIL_CONFIG["use_ssl"]
            self.batch_size = FETCH_CONFIG["batch_size"]
        except Exception as e:
            logger.error(f"Error reading configuration: {e}")
            raise

        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize the database with required tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create emails table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT UNIQUE,
                sender TEXT,
                recipient TEXT,
                subject TEXT,
                date_received TIMESTAMP,
                body_text TEXT,
                body_html TEXT,
                body_html_cleaned TEXT,
                raw_email BLOB,
                processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create attachments table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id INTEGER,
                filename TEXT,
                content_type TEXT,
                data BLOB,
                FOREIGN KEY (email_id) REFERENCES emails (id)
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def connect_to_mail_server(self):
        """Connect to the mail server using the configured settings."""
        try:
            print(f"Attempting to connect to {self.imap_server}:{self.imap_port}...")
            if self.use_ssl:
                mail = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            else:
                mail = imaplib.IMAP4(self.imap_server, self.imap_port)
                
            print(f"Connected to server, attempting to login as {self.email_address}...")
            mail.login(self.email_address, self.password)
            logger.info(f"Successfully connected to {self.imap_server}")
            return mail
        except Exception as e:
            error_msg = f"Failed to connect to mail server: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            raise
    
    def get_emails_since(self, since_date, mailbox="INBOX"):
        """
        Retrieve emails from the specified mailbox since the given date.
        
        Args:
            since_date: datetime object representing the cutoff date
            mailbox: the mailbox to search (default is "INBOX")
        """
        try:
            # Format date for IMAP query (DD-MMM-YYYY format)
            date_str = since_date.strftime("%d-%b-%Y")
            
            # Connect to mail server
            mail = self.connect_to_mail_server()
            
            # Select the mailbox
            status, messages = mail.select(mailbox)
            if status != 'OK':
                logger.error(f"Failed to select mailbox {mailbox}: {messages}")
                mail.logout()
                return
                
            # Search for emails since the specified date
            status, data = mail.search(None, f'(SINCE {date_str})')
            if status != 'OK':
                logger.error(f"Failed to search for emails: {data}")
                mail.logout()
                return
                
            # Get list of email IDs
            email_ids = data[0].split()
            logger.info(f"Found {len(email_ids)} emails since {date_str}")
            
            # Process emails in batches
            for i in range(0, len(email_ids), self.batch_size):
                batch = email_ids[i:i+self.batch_size]
                for email_id in batch:
                    self._process_email(mail, email_id)
                
                logger.info(f"Processed batch {i//self.batch_size + 1}/{(len(email_ids)-1)//self.batch_size + 1}")
            
            mail.logout()
            logger.info("Email fetch and processing completed")
        except Exception as e:
            logger.error(f"Error retrieving emails: {e}")
            raise
    
    def _process_email(self, mail, email_id):
        """Process a single email and store it in the database."""
        try:
            # Fetch the email content
            status, msg_data = mail.fetch(email_id, '(RFC822)')
            if status != 'OK':
                logger.error(f"Failed to fetch email {email_id}: {msg_data}")
                return
                
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            
            # Extract basic email information
            message_id = msg.get('Message-ID', '')
            
            # Skip if this email is already in the database
            if self._email_exists(message_id):
                logger.debug(f"Email {message_id} already exists in database, skipping")
                return
                
            sender = msg.get('From', '')
            recipient = msg.get('To', '')
            subject = self._decode_header(msg.get('Subject', ''))
            date_str = msg.get('Date', '')
            
            # Parse date
            date_received = None
            if date_str:
                try:
                    # Parse the date string and convert to datetime
                    date_tuple = email.utils.parsedate_tz(date_str)
                    if date_tuple:
                        date_received = datetime.datetime.fromtimestamp(
                            email.utils.mktime_tz(date_tuple)
                        )
                except Exception as e:
                    logger.warning(f"Could not parse date {date_str}: {e}")
            
            # Extract body content
            body_text = ""
            body_html = ""
            body_html_cleaned = ""
            attachments = []
            
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    
                    # Skip any multipart containers
                    if content_type == "multipart/alternative" or content_type == "multipart/mixed":
                        continue
                        
                    # Handle attachments
                    if "attachment" in content_disposition:
                        filename = part.get_filename()
                        if filename:
                            filename = self._decode_header(filename)
                            attachments.append({
                                'filename': filename,
                                'content_type': content_type,
                                'data': part.get_payload(decode=True)
                            })
                            continue
                    
                    # Get the payload
                    payload = part.get_payload(decode=True)
                    if payload is None:
                        continue
                        
                    # Decode the payload if it's text
                    charset = part.get_content_charset()
                    if charset:
                        try:
                            payload = payload.decode(charset)
                        except UnicodeDecodeError:
                            payload = payload.decode(charset, 'replace')
                    else:
                        try:
                            payload = payload.decode()
                        except UnicodeDecodeError:
                            payload = payload.decode('utf-8', 'replace')
                    
                    # Store the content based on type
                    if content_type == "text/plain":
                        body_text = payload
                    elif content_type == "text/html":
                        body_html = payload
                        body_html_cleaned = self._remove_html_tags(payload)
            else:
                # Not multipart - get the content directly
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset()
                    if charset:
                        try:
                            payload = payload.decode(charset)
                        except UnicodeDecodeError:
                            payload = payload.decode(charset, 'replace')
                    else:
                        try:
                            payload = payload.decode()
                        except UnicodeDecodeError:
                            payload = payload.decode('utf-8', 'replace')
                            
                    content_type = msg.get_content_type()
                    if content_type == "text/plain":
                        body_text = payload
                    elif content_type == "text/html":
                        body_html = payload
                        body_html_cleaned = self._remove_html_tags(payload)
            
            # Store email in database
            self._store_email(
                message_id, sender, recipient, subject, date_received, 
                body_text, body_html, body_html_cleaned, raw_email, attachments
            )
        except Exception as e:
            logger.error(f"Error processing email {email_id}: {e}")
    
    def _email_exists(self, message_id):
        """Check if an email with the given message_id already exists in the database."""
        if not message_id:
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM emails WHERE message_id = ?", (message_id,))
            result = cursor.fetchone()
            conn.close()
            return result is not None
        except Exception as e:
            logger.error(f"Error checking if email exists: {e}")
            return False
    
    def _store_email(self, message_id, sender, recipient, subject, date_received, 
                    body_text, body_html, body_html_cleaned, raw_email, attachments):
        """Store email and its attachments in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert email
            cursor.execute('''
            INSERT INTO emails 
            (message_id, sender, recipient, subject, date_received, body_text, body_html, body_html_cleaned, raw_email)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                message_id, sender, recipient, subject, 
                date_received, body_text, body_html, body_html_cleaned, raw_email
            ))
            
            email_id = cursor.lastrowid
            
            # Insert attachments
            for attachment in attachments:
                cursor.execute('''
                INSERT INTO attachments
                (email_id, filename, content_type, data)
                VALUES (?, ?, ?, ?)
                ''', (
                    email_id, attachment['filename'], 
                    attachment['content_type'], attachment['data']
                ))
            
            conn.commit()
            conn.close()
            logger.debug(f"Stored email with ID {email_id} in database")
        except sqlite3.IntegrityError as e:
            logger.warning(f"Skipping duplicate email: {e}")
        except Exception as e:
            logger.error(f"Error storing email in database: {e}")
    
    def _decode_header(self, header):
        """Decode encoded email headers."""
        if not header:
            return ""
            
        try:
            decoded_headers = decode_header(header)
            header_parts = []
            
            for content, encoding in decoded_headers:
                if isinstance(content, bytes):
                    if encoding:
                        header_parts.append(content.decode(encoding))
                    else:
                        try:
                            header_parts.append(content.decode())
                        except UnicodeDecodeError:
                            header_parts.append(content.decode('utf-8', 'replace'))
                else:
                    header_parts.append(str(content))
                    
            return " ".join(header_parts)
        except Exception as e:
            logger.warning(f"Error decoding header: {e}")
            return header

    def create_draft_reply(self, original_message: Dict, response_text: str) -> None:
        """
        Create a draft reply using IMAP.
        
        Args:
            original_message: Dictionary containing original email details
            response_text: Generated response text
        """
        try:
            # Connect to mail server if not already connected
            mail = self.connect_to_mail_server()
            
            # Create message
            msg = MIMEText(response_text)
            msg['From'] = self.email_address
            msg['To'] = original_message['sender']
            msg['Subject'] = f"Re: {original_message['subject']}"
            msg['Date'] = formatdate(localtime=True)
            msg['In-Reply-To'] = original_message['message_id']
            msg['References'] = original_message['message_id']
            
            # Select the Drafts folder
            status, _ = mail.select('[Gmail]/Drafts')  # For Gmail. Adjust folder name if using different provider
            if status != 'OK':
                logger.error("Could not select Drafts folder")
                return None
                
            # Append the message to Drafts folder with Draft flag
            status, [msg_id] = mail.append('[Gmail]/Drafts', '\\Draft', imaplib.Time2Internaldate(time.time()), msg.as_bytes())
            
            if status == 'OK':
                logger.info("Draft created successfully")
                return msg_id
            else:
                logger.error(f"Failed to create draft: {status}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating draft: {str(e)}")
            return None
        finally:
            try:
                mail.close()
            except:
                pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Email retrieval service')
    parser.add_argument('--since', help='Fetch emails since this date (format: YYYY-MM-DD)')
    parser.add_argument('--mailbox', default='INBOX', help='Mailbox to fetch from')
    
    args = parser.parse_args()
    
    try:
        print("Initializing email service...")
        # Initialize service
        service = EmailService()
        
        if args.since:
            since_date = datetime.datetime.strptime(args.since, "%Y-%m-%d")
        else:
            # Default: fetch emails from the last 24 hours
            since_date = datetime.datetime.now() - timedelta(minutes=6)
            
        print(f"Fetching emails since {since_date} from {args.mailbox}...")
        service.get_emails_since(since_date, args.mailbox)
        print("Email fetch completed successfully")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.error(f"Fatal error: {e}")