from typing import List, Dict, Any, Optional, TypedDict, Annotated
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
# from langgraph.prebuilt import ToolExecutor
# from langchain_core.tools import tool
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MAX_RETRIEVAL_DOCS,
    MODEL_NAME,
    TEMPERATURE,
    INDEX_NAME,
    API_KEYS, 
    RULES
)

from prompts.langgraph_agent_prompts import (
    REPLY_DECISION_PROMPT,
    SEARCH_QUERY_PROMPT,
    EMAIL_RESPONSE_PROMPT
)

# Pydantic models for structured responses
class ReplyDecision(BaseModel):
    decision: bool = Field(description="Whether to reply to the email (true/false)")
    reasoning: str = Field(description="Explanation for the decision")

class SearchQuery(BaseModel):
    search_query: Optional[str] = Field(description="Search query to find relevant information, or null if no search needed")
    reasoning: str = Field(description="Explanation for the search query decision")

class EmailResponse(BaseModel):
    response: str = Field(description="The generated email response")
    reasoning: str = Field(description="Explanation for the response generation")

# Define the state structure
class AgentState(TypedDict):
    email_content: str
    should_reply: Optional[bool]
    search_query: Optional[str]
    retrieved_context: Optional[str]
    final_response: Optional[str]
    reasoning: List[str]

class LangGraphEmailAgent:
    def __init__(self, index_name: str = INDEX_NAME, rules: Dict[str, List[str]] = None):
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
        
        # Set rules as instance variable
        self.rules = rules or RULES
        
        # Initialize output parsers
        self.reply_parser = PydanticOutputParser(pydantic_object=ReplyDecision)
        self.search_parser = PydanticOutputParser(pydantic_object=SearchQuery)
        self.email_parser = PydanticOutputParser(pydantic_object=EmailResponse)
        
        # Create the graph
        self.graph = self._create_graph()
    
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
    
    def reply_decide(self, state: AgentState) -> AgentState:
        """Node 1: Decide whether to reply to the email."""
        prompt_template = PromptTemplate(
            template=REPLY_DECISION_PROMPT,
            input_variables=["email_content", "rules"],
            partial_variables={"format_instructions": self.reply_parser.get_format_instructions()}
        )
        
        prompt = prompt_template.format(
            email_content=state['email_content'],
            rules=self.rules['emailResponseRules']
        )
        
        try:
            response = self.llm.invoke(prompt)
            decision_data = self.reply_parser.parse(response.content)
        except Exception as e:
            print(f"Error parsing reply decision: {e}")
            decision_data = ReplyDecision(decision=False, reasoning=f"Error parsing response: {str(e)}")
        
        return {
            **state,
            "should_reply": decision_data.decision,
            "reasoning": state["reasoning"] + [f"Reply decision: {decision_data.reasoning}"]
        }
    
    def search_query_generator(self, state: AgentState) -> AgentState:
        """Node 2: Generate search query"""
        if not state.get("should_reply"):
            return {
                **state,
                "search_query": None,
                "reasoning": state["reasoning"] + ["No search query needed - no reply required"]
            }
        
        prompt_template = PromptTemplate(
            template=SEARCH_QUERY_PROMPT,
            input_variables=["email_content"],
            partial_variables={"format_instructions": self.search_parser.get_format_instructions()}
        )
        
        prompt = prompt_template.format(email_content=state['email_content'])
        
        try:
            response = self.llm.invoke(prompt)
            search_data = self.search_parser.parse(response.content)
        except Exception as e:
            print(f"Error parsing search query: {e}")
            search_data = SearchQuery(search_query=None, reasoning=f"Error parsing response: {str(e)}")
        
        return {
            **state,
            "search_query": search_data.search_query,
            "reasoning": state["reasoning"] + [f"Search query generation: {search_data.reasoning}"]
        }
    
    def retrieve_info(self, state: AgentState) -> AgentState:
        """Node 3: Retrieve relevant information from the knowledge base."""
        if not state.get("search_query"):
            return {
                **state,
                "retrieved_context": "No search query provided",
                "reasoning": state["reasoning"] + ["No information retrieval needed"]
            }
        
        context = self._get_relevant_context(state["search_query"])
        
        return {
            **state,
            "retrieved_context": context if context else "No relevant information found",
            "reasoning": state["reasoning"] + [f"Retrieved information for query: {state['search_query']}"]
        }
    
    def generate_mail(self, state: AgentState) -> AgentState:
        """Node 4: Generate the final email response."""
        if not state.get("should_reply"):
            return {
                **state,
                "final_response": "No reply needed based on the analysis.",
                "reasoning": state["reasoning"] + ["No reply generated - not needed"]
            }
        
        prompt_template = PromptTemplate(
            template=EMAIL_RESPONSE_PROMPT,
            input_variables=["email_content", "retrieved_context", "rules"],
            partial_variables={"format_instructions": self.email_parser.get_format_instructions()}
        )
        
        prompt = prompt_template.format(
            email_content=state['email_content'],
            retrieved_context=state.get('retrieved_context', 'No context available'),
            rules=self.rules['emailResponseRules']
        )
        
        try:
            response = self.llm.invoke(prompt)
            email_data = self.email_parser.parse(response.content)
        except Exception as e:
            print(f"Error parsing email response: {e}")
            email_data = EmailResponse(
                response="Unable to generate response due to parsing error.",
                reasoning=f"Error parsing response: {str(e)}"
            )
        
        return {
            **state,
            "final_response": email_data.response,
            "reasoning": state["reasoning"] + [f"Email response generated: {email_data.reasoning}"]
        }
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("reply_decide", self.reply_decide)
        workflow.add_node("search_query_generator", self.search_query_generator)
        workflow.add_node("retrieve_info", self.retrieve_info)
        workflow.add_node("generate_mail", self.generate_mail)
        
        # Define the flow
        workflow.set_entry_point("reply_decide")
        
        # Conditional flow from reply_decide
        def route_after_reply_decision(state: AgentState) -> str:
            if state.get("should_reply"):
                return "search_query_generator"
            else:
                return END
        
        workflow.add_conditional_edges(
            "reply_decide",
            route_after_reply_decision,
            {
                "search_query_generator": "search_query_generator",
                END: END
            }
        )
        
        # Conditional flow from search_query_generator
        def route_after_search_query(state: AgentState) -> str:
            if state.get("search_query"):
                return "retrieve_info"
            else:
                return "generate_mail"
        
        workflow.add_conditional_edges(
            "search_query_generator",
            route_after_search_query,
            {
                "retrieve_info": "retrieve_info",
                "generate_mail": "generate_mail"
            }
        )
        
        workflow.add_edge("retrieve_info", "generate_mail")
        workflow.add_edge("generate_mail", END)
        
        return workflow.compile()
    
    def generate_response(self, email_content: str) -> str:
        """Generate a response using the LangGraph workflow."""
        # Initialize the state
        initial_state = AgentState(
            email_content=email_content,
            should_reply=None,
            search_query=None,
            retrieved_context=None,
            final_response=None,
            reasoning=[]
        )
        
        # Execute the graph
        result = self.graph.invoke(initial_state)
        
        # Check if we ended early (no reply needed)
        if result.get("should_reply") == False:
            return "No reply needed based on the analysis."
        
        # Return the final response
        return result.get("final_response", "Unable to generate response")
    
    def print_graph_structure(self):
        """Print the graph structure using LangGraph's built-in Mermaid generation."""
        print("\nWorkflow Graph Structure:")
        mermaid = self.graph.get_graph().draw_mermaid()
        print(mermaid)

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = LangGraphEmailAgent()
    
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
    
    # Test responses
    for email in emails:
        print("\nProcessing email:")
        print(email)
        print("\nGenerated Response:")
        print('Final email output:', agent.generate_response(email))
        print("-" * 80)
        #agent.print_graph_structure() 