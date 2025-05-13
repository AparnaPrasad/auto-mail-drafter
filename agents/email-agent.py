from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_RETRIEVAL_DOCS,
    MODEL_NAME,
    TEMPERATURE,
    INDEX_NAME,
    API_KEYS
)

class EmailRAGAgent:
    def __init__(self, index_name: str = INDEX_NAME):
        # Initialize OpenAI components
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            api_key=API_KEYS["openai"]
        )
        self.embeddings = OpenAIEmbeddings(
            api_key=API_KEYS["openai"]
        )
        
        # Initialize Pinecone
        self.vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=self.embeddings,
            pinecone_api_key=API_KEYS["pinecone"]
        )
    
    def _get_relevant_context(self, query: str) -> Optional[str]:
        """Retrieve relevant context from the vector store."""
        try:
            docs = self.vector_store.similarity_search(
                query,
                k=MAX_RETRIEVAL_DOCS
            )
            return "\n".join(doc.page_content for doc in docs)
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return None
    
    def generate_response(self, email_content: str, rules: Dict[str, List[str]]) -> str:
        """Generate a response using ReAct prompting."""
        react_prompt = f"""
            You are an email response assistant using the ReAct (Reasoning + Acting) framework. 
            Your goal is to determine whether and how to respond to an incoming email using reasoning and available actions.

            You can use the following actions:
            - search_knowledge_base(query: str): Search the internal knowledge base for relevant information.
            - generate_response(email: str, context: str): Generate the final email response based on the given email and any retrieved context.

            All actions must be written as executable calls, e.g.:
            ACT: search_knowledge_base("query here")
            ...
            ACT: generate_response(email="{email_content}", context="retrieved context here")

            Follow this loop:
            1. THINK: Read the email carefully. Decide if a reply is necessary. Follow the rules {rules["reactRules"]} to decide if you need to respond. If a reply is needed, identify what information is needed. Do not make up any information. You have access to the company's knowledge base. 
            2. ACT: Take an action based on your reasoning. For example:
            - If reply is not needed, decide it's a "no-reply."
            - If information is needed, call `search_knowledge_base(query: str)`.
            3. OBSERVE: Review the results of your action.
            4. THINK: Decide if further action is needed (e.g., another search).
            5. ACT: Repeat search using search_knowledge_base(query: str) or proceed to step 6 that is response generation.
            6. RESPOND: If a reply is warranted, you **must** call `generate_response(email: str, context: str)` with the email and relevant context.
            Do not summarize or describe your response. Just **call the function**.
            Email to analyze:
            {email_content}

            Begin by reasoning through the first THINK step.
            """

        # Initialize conversation
        messages = [{"role": "system", "content": react_prompt}]
        
        while True:
            # Get the next action from the model
            response = self.llm.invoke(messages)
            messages.append({"role": "assistant", "content": response.content})
            print('LLM response:', response.content)
            # Parse the model's response
            content = response.content.lower()
            
            if "search_knowledge_base" in content:
                print('Searching knowledge base...')
                # Extract the query from the action
                query_start = content.find("search_knowledge_base(") + len("search_knowledge_base(")
                query_end = content.find(")", query_start)
                query = content[query_start:query_end].strip()
                
                # Get context
                context = self._get_relevant_context(query)
                
                # Add observation to messages
                messages.append({
                    "role": "system",
                    "content": f"Knowledge base search results for '{query}':\n{context if context else 'No relevant information found.'}"
                })
                
            elif "generate_response" in content:
                print('Generating response...')
                # Extract email and context from the action

                email_start = content.find("email=") + len("email=")
                email_end = content.find(",", email_start)
                context_start = content.find("context=") + len("context=")
                context_end = content.find(")", context_start)
                
                email = content[email_start:email_end].strip()
                context = content[context_start:context_end].strip()
                
                # Generate final response
                final_prompt = f"""
                You are a customer support agent. That replies to emails. Given an email and context (retrieved from the company's knowledge base).
                Generate a professional email responding to the email using the following information:
                
                Email to respond to:
                {email}
                
                Context (if available):
                {context}
                
                Write a professional and helpful email response.
                Note, context is not sent by the customer. It is from the company's knowledge base.

                Here are some rules to follow:
                {rules["emailResponseRules"]}
                """
                
                final_response = self.llm.invoke(final_prompt)
                return final_response.content
            
            elif "respond:" in content:
                print('Responding...')
                # Extract the response after "respond:"
                response_text = content.split("respond:", 1)[1].strip()
                return response_text
            
            # Add a limit to prevent infinite loops
            if len(messages) > 10:
                return "I apologize, but I'm having trouble generating a response. Please try rephrasing your email."
    
    def add_documents(self, documents: List[str]):
        """Add documents to Pinecone."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        texts = text_splitter.split_text("\n".join(documents))
        
        # Add documents to Pinecone
        #PineconeVectorStore.from_documents(documents=texts, embedding=self.embeddings, index_name=INDEX_NAME)
        self.vector_store.add_texts(texts)

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = EmailRAGAgent()
    
    # Add some example documents
    documents = [
        "Our company policy states that refunds must be requested within 30 days of purchase.",
        "Technical support is available 24/7 through our help desk portal.",
        "All premium features are included in the enterprise subscription plan.",
        "To access premium features, users need to log in to their account and navigate to the 'Premium' section.",
        "Refund requests should be submitted through the customer portal with the order number and reason for refund."
    ]
    # agent.add_documents(documents)
    
    # Example emails to test different scenarios
    emails = [
        # Email requiring context
        """
        Dear Support Team,
        
        I purchased your product last month and I'm having some issues with the premium features.
        Can you help me understand how to access these features and what my options are for getting a refund if I'm not satisfied?
        
        Best regards,
        John Doe
        """,
        
        # Email not requiring context
        """
        Hi there,
        
        Thank you for your prompt response to my previous inquiry. I appreciate your help.
        
        Best regards,
        Jane Smith
        """
    ]

    rules = {
    "emailResponseRules": [
        "1. Keep the tone friendly and professional.",
        "2. Keep the response concise and to the point.",
        "3. Your name is Aparna Prasad, CEO of the company.",
        "4. The company name is Acme Corp.",
        "5. The company address is 123 Main St, Anytown, USA 12345.",
        "6. The company phone number is 123-456-7890.",
        "7. The company email is support@acme.com.",
        "8. The company website is https://www.acme.com."
    ],
    "reactRules": [
        "1. Do not respond to emails that are not relevant to the company.",
        "2. If the email is not relevant, do not respond.",
        "3. Do not summarize or describe your response. Just **call the function**.",
        "4. Do not respond to no-reply or alert emails.",
        "5. If its a thank you email, respond."
    ]
    }
    
    # Test responses
    for email in emails:
        print("\nProcessing email:")
        print(email)
        print("\nGenerated Response:")
        print('D=Fianl email output:::', agent.generate_response(email, rules))
        print("-" * 80)
