import sys
import os
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the agents after adding to sys.path
from agents.email_agent import EmailRAGAgent
from agents.langgraph_email_agent import LangGraphEmailAgent
from config import RULES, EVALUATION_CONFIG

@dataclass
class TestCase:
    """Represents a test case for email response evaluation."""
    id: str
    email_content: str
    expected_reply_needed: bool
    expected_context_needed: bool
    expected_search_queries: List[str]  # Expected search terms
    category: str
    description: str
    should_retrieve_info: bool  # Whether agent should use knowledge base

@dataclass
class ToolUsage:
    """Tracks tool usage during agent execution."""
    search_calls: List[str] = None  # List of search queries made
    search_results: List[str] = None  # Results from searches
    tool_used: bool = False
    tool_usage_correct: bool = True  # Whether tool was used appropriately

@dataclass
class AgentResponse:
    """Represents a response from an agent with detailed tracking."""
    response: str
    processing_time: float
    tool_usage: ToolUsage
    reasoning: List[str] = None
    search_query: str = None
    retrieved_context: str = None

@dataclass
class EvaluationResult:
    """Represents evaluation results for a test case."""
    test_case: TestCase
    email_agent_response: AgentResponse
    langgraph_agent_response: AgentResponse
    response_similarity: float = 0.0
    email_agent_score: float = 0.0
    langgraph_agent_score: float = 0.0
    tool_usage_comparison: Dict[str, Any] = None

class EmailAgentEvaluator:
    def __init__(self):
        self.email_agent = EmailRAGAgent()
        self.langgraph_agent = LangGraphEmailAgent()
        self.test_cases = self._create_test_cases()
        
        # Load configuration from EVALUATION_CONFIG
        self.evaluation_timeout = EVALUATION_CONFIG["timeout"]
        self.max_retries = EVALUATION_CONFIG["max_retries"]
        self.save_detailed_logs = EVALUATION_CONFIG["save_detailed_logs"]
        self.output_dir = EVALUATION_CONFIG["output_dir"]
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Scoring weights from EVALUATION_CONFIG
        self.tool_usage_weight = EVALUATION_CONFIG["scoring_weights"]["tool_usage"]
        self.response_quality_weight = EVALUATION_CONFIG["scoring_weights"]["response_quality"]
        self.processing_time_weight = EVALUATION_CONFIG["scoring_weights"]["processing_time"]
        self.similarity_weight = EVALUATION_CONFIG["scoring_weights"]["similarity"]
    
    def _create_test_cases(self) -> List[TestCase]:
        """Create a single test case to check tool usage."""
        return [
            # Test Case: Should retrieve pricing information
            TestCase(
                id="PRICING001",
                email_content="""Hello,
                
I'm interested in your premium subscription. What are the current pricing plans and what features are included in each tier?

Thanks,
Alex""",
                expected_reply_needed=True,
                expected_context_needed=True,
                expected_search_queries=["pricing", "premium", "subscription", "plans", "features"],
                category="pricing_inquiry",
                description="Pricing inquiry requiring knowledge base lookup",
                should_retrieve_info=True
            )
        ]
    
    def _extract_search_queries_from_langgraph_agent(self, state: Dict) -> List[str]:
        """Extract search queries from LangGraph Agent's state."""
        queries = []
        
        # Check if search query was generated
        if state.get("search_query"):
            queries.append(state["search_query"])
        
        return queries
    
    def _evaluate_tool_usage(self, test_case: TestCase, actual_queries: List[str], 
                           should_retrieve: bool, retrieved_context: str = None) -> ToolUsage:
        """Evaluate whether the agent used tools appropriately."""
        tool_usage = ToolUsage(
            search_calls=actual_queries,
            search_results=[retrieved_context] if retrieved_context else [],
            tool_used=len(actual_queries) > 0
        )
        
        # Check if tool usage was correct
        if should_retrieve and not tool_usage.tool_used:
            tool_usage.tool_usage_correct = False  # Should have used tools but didn't
        elif not should_retrieve and tool_usage.tool_used:
            tool_usage.tool_usage_correct = False  # Used tools when not needed
        else:
            tool_usage.tool_usage_correct = True  # Tool usage was appropriate
        
        return tool_usage
    
    def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """Calculate similarity between two responses."""
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _evaluate_response_quality(self, response: str, test_case: TestCase, 
                                 tool_usage: ToolUsage) -> float:
        """Evaluate response quality considering tool usage."""
        score = 0.0
        response_lower = response.lower()
        
        # Base appropriateness score
        if test_case.expected_reply_needed:
            if "no reply" in response_lower or "not needed" in response_lower:
                score -= 0.5
            else:
                score += 0.3
        else:
            if "no reply" in response_lower or "not needed" in response_lower:
                score += 0.3
            else:
                score -= 0.2
        
        # Tool usage score (major component) - use environment weight
        if tool_usage.tool_usage_correct:
            score += self.tool_usage_weight  # Use configured weight
        else:
            score -= self.tool_usage_weight  # Use configured weight
        
        # Professional tone
        professional_indicators = ["thank you", "appreciate", "regards", "best regards", "sincerely"]
        if any(indicator in response_lower for indicator in professional_indicators):
            score += 0.2
        
        # Helpfulness
        helpful_indicators = ["help", "assist", "support", "guide", "explain", "provide"]
        if any(indicator in response_lower for indicator in helpful_indicators):
            score += 0.2
        
        # Length check
        word_count = len(response.split())
        if 10 <= word_count <= 200:
            score += 0.1
        elif word_count < 5:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _test_email_agent_with_tool_tracking(self, email_content: str, rules: Dict) -> Tuple[str, List[str], float]:
        """Test Email Agent with tool usage tracking during ReAct process."""
        start_time = time.time()
        
        # Generate response and get intermediate steps
        response, intermediate_steps = self.email_agent.generate_response(email_content, rules)
        processing_time = time.time() - start_time
        
        # Extract search queries from intermediate steps
        search_queries = self._extract_search_queries_from_intermediate_steps(intermediate_steps)
        
        return response, search_queries, processing_time
    
    def _extract_search_queries_from_intermediate_steps(self, messages: List[Dict]) -> List[str]:
        """Extract search queries from the intermediate ReAct steps."""
        queries = []
        
        for message in messages:
            if message.get("role") == "assistant":
                content = message.get("content", "").lower()
                
                # Look for search_knowledge_base calls
                if "search_knowledge_base" in content:
                    # Extract the query from the action
                    query_start = content.find("search_knowledge_base(") + len("search_knowledge_base(")
                    query_end = content.find(")", query_start)
                    if query_start > len("search_knowledge_base(") - 1 and query_end > query_start:
                        query = content[query_start:query_end].strip()
                        # Clean up quotes if present
                        query = query.strip('"\'')
                        if query:
                            queries.append(query)
                            print(f"EXTRACTED SEARCH: {query}")
        
        return list(set(queries))  # Remove duplicates
    
    def run_single_test(self, test_case: TestCase) -> EvaluationResult:
        """Run a single test case with detailed tool usage tracking."""
        print(f"\nRunning test case: {test_case.id} - {test_case.description}")
        print(f"Expected to retrieve info: {test_case.should_retrieve_info}")
        print(f"Expected search queries: {test_case.expected_search_queries}")
        
        # Test Email Agent
        print("\nTesting Email Agent...")
        start_time = time.time()
        
        try:
            email_response, email_queries, email_processing_time = self._test_email_agent_with_tool_tracking(test_case.email_content, RULES)
            print(f"Email Agent search queries: {email_queries}")
            
            # Evaluate tool usage for Email Agent
            email_tool_usage = self._evaluate_tool_usage(
                test_case, email_queries, test_case.should_retrieve_info
            )
            
            email_agent_response = AgentResponse(
                response=email_response,
                processing_time=email_processing_time,
                tool_usage=email_tool_usage
            )
            
        except Exception as e:
            print(f"Email Agent Error: {e}")
            email_agent_response = AgentResponse(
                response=f"Error: {str(e)}",
                processing_time=time.time() - start_time,
                tool_usage=ToolUsage(tool_usage_correct=False)
            )
        
        # Test LangGraph Agent
        print("\nTesting LangGraph Agent...")
        start_time = time.time()
        
        try:
            # We need to capture the state to analyze tool usage
            initial_state = {
                "email_content": test_case.email_content,
                "should_reply": None,
                "search_query": None,
                "retrieved_context": None,
                "final_response": None,
                "reasoning": []
            }
            
            # Execute the graph and capture state
            result = self.langgraph_agent.graph.invoke(initial_state)
            langgraph_processing_time = time.time() - start_time
            
            # Extract search queries from LangGraph Agent state
            langgraph_queries = self._extract_search_queries_from_langgraph_agent(result)
            print(f"LangGraph Agent search queries: {langgraph_queries}")
            
            # Evaluate tool usage for LangGraph Agent
            langgraph_tool_usage = self._evaluate_tool_usage(
                test_case, langgraph_queries, test_case.should_retrieve_info,
                result.get("retrieved_context")
            )
            
            langgraph_agent_response = AgentResponse(
                response=result.get("final_response", "No response generated"),
                processing_time=langgraph_processing_time,
                tool_usage=langgraph_tool_usage,
                reasoning=result.get("reasoning", []),
                search_query=result.get("search_query"),
                retrieved_context=result.get("retrieved_context")
            )
            
        except Exception as e:
            print(f"LangGraph Agent Error: {e}")
            langgraph_agent_response = AgentResponse(
                response=f"Error: {str(e)}",
                processing_time=time.time() - start_time,
                tool_usage=ToolUsage(tool_usage_correct=False)
            )
        
        # Calculate similarity and scores
        similarity = self._calculate_response_similarity(
            email_agent_response.response, 
            langgraph_agent_response.response
        )
        
        email_score = self._evaluate_response_quality(
            email_agent_response.response, test_case, email_agent_response.tool_usage
        )
        langgraph_score = self._evaluate_response_quality(
            langgraph_agent_response.response, test_case, langgraph_agent_response.tool_usage
        )
        
        # Tool usage comparison
        tool_usage_comparison = {
            "email_agent": {
                "tool_used": email_agent_response.tool_usage.tool_used,
                "search_queries": email_agent_response.tool_usage.search_calls,
                "tool_usage_correct": email_agent_response.tool_usage.tool_usage_correct
            },
            "langgraph_agent": {
                "tool_used": langgraph_agent_response.tool_usage.tool_used,
                "search_queries": langgraph_agent_response.tool_usage.search_calls,
                "tool_usage_correct": langgraph_agent_response.tool_usage.tool_usage_correct
            }
        }
        
        return EvaluationResult(
            test_case=test_case,
            email_agent_response=email_agent_response,
            langgraph_agent_response=langgraph_agent_response,
            response_similarity=similarity,
            email_agent_score=email_score,
            langgraph_agent_score=langgraph_score,
            tool_usage_comparison=tool_usage_comparison
        )
    
    def run_evaluation(self) -> List[EvaluationResult]:
        """Run evaluation on all test cases."""
        results = []
        
        print("Starting Email Agent Evaluation with Tool Usage Analysis...")
        print(f"Running the same test case 10 times for consistency analysis")
        
        # Run the same test case 10 times
        test_case = self.test_cases[0]  # Get the single test case
        
        for run_number in range(1, 11):
            print(f"\n{'='*80}")
            print(f"Run {run_number}/10: {test_case.id}")
            print(f"Category: {test_case.category}")
            print(f"Description: {test_case.description}")
            print(f"Should retrieve info: {test_case.should_retrieve_info}")
            print(f"{'='*80}")
            
            result = self.run_single_test(test_case)
            results.append(result)
            
            # Print detailed results for each run
            print(f"\nResults for Run {run_number}:")
            print(f"Email Agent Response: {result.email_agent_response.response[:200]}...")
            print(f"LangGraph Agent Response: {result.langgraph_agent_response.response[:200]}...")
            print(f"Response Similarity: {result.response_similarity:.3f}")
            print(f"Email Agent Score: {result.email_agent_score:.3f}")
            print(f"LangGraph Agent Score: {result.langgraph_agent_score:.3f}")
            print(f"Email Agent Time: {result.email_agent_response.processing_time:.2f}s")
            print(f"LangGraph Agent Time: {result.langgraph_agent_response.processing_time:.2f}s")
            
            # Tool usage details
            print(f"\nTool Usage Analysis (Run {run_number}):")
            print(f"Email Agent - Tool Used: {result.email_agent_response.tool_usage.tool_used}")
            print(f"Email Agent - Search Queries: {result.email_agent_response.tool_usage.search_calls}")
            print(f"Email Agent - Tool Usage Correct: {result.email_agent_response.tool_usage.tool_usage_correct}")
            print(f"LangGraph Agent - Tool Used: {result.langgraph_agent_response.tool_usage.tool_used}")
            print(f"LangGraph Agent - Search Queries: {result.langgraph_agent_response.tool_usage.search_calls}")
            print(f"LangGraph Agent - Tool Usage Correct: {result.langgraph_agent_response.tool_usage.tool_usage_correct}")
        
        return results
    
    def _analyze_consistency(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze consistency across multiple runs of the same test case."""
        if len(results) < 2:
            return {}
        
        # Tool usage consistency
        email_tool_usage_consistency = sum(1 for r in results if r.email_agent_response.tool_usage.tool_used) / len(results)
        langgraph_tool_usage_consistency = sum(1 for r in results if r.langgraph_agent_response.tool_usage.tool_used) / len(results)
        
        email_correct_tool_usage_consistency = sum(1 for r in results if r.email_agent_response.tool_usage.tool_usage_correct) / len(results)
        langgraph_correct_tool_usage_consistency = sum(1 for r in results if r.langgraph_agent_response.tool_usage.tool_usage_correct) / len(results)
        
        # Score consistency (standard deviation)
        email_scores = [r.email_agent_score for r in results]
        langgraph_scores = [r.langgraph_agent_score for r in results]
        
        import statistics
        email_score_std = statistics.stdev(email_scores) if len(email_scores) > 1 else 0
        langgraph_score_std = statistics.stdev(langgraph_scores) if len(langgraph_scores) > 1 else 0
        
        # Processing time consistency
        email_times = [r.email_agent_response.processing_time for r in results]
        langgraph_times = [r.langgraph_agent_response.processing_time for r in results]
        
        email_time_std = statistics.stdev(email_times) if len(email_times) > 1 else 0
        langgraph_time_std = statistics.stdev(langgraph_times) if len(langgraph_times) > 1 else 0
        
        # Winner consistency
        email_wins = sum(1 for r in results if r.email_agent_score > r.langgraph_agent_score)
        langgraph_wins = sum(1 for r in results if r.langgraph_agent_score > r.email_agent_score)
        ties = len(results) - email_wins - langgraph_wins
        
        return {
            "tool_usage_consistency": {
                "email_agent_tool_usage_rate": email_tool_usage_consistency,
                "langgraph_agent_tool_usage_rate": langgraph_tool_usage_consistency,
                "email_agent_correct_tool_usage_rate": email_correct_tool_usage_consistency,
                "langgraph_agent_correct_tool_usage_rate": langgraph_correct_tool_usage_consistency
            },
            "score_consistency": {
                "email_agent_score_std": email_score_std,
                "langgraph_agent_score_std": langgraph_score_std,
                "email_agent_score_range": (min(email_scores), max(email_scores)),
                "langgraph_agent_score_range": (min(langgraph_scores), max(langgraph_scores))
            },
            "time_consistency": {
                "email_agent_time_std": email_time_std,
                "langgraph_agent_time_std": langgraph_time_std,
                "email_agent_time_range": (min(email_times), max(email_times)),
                "langgraph_agent_time_range": (min(langgraph_times), max(langgraph_times))
            },
            "winner_consistency": {
                "email_agent_wins": email_wins,
                "langgraph_agent_wins": langgraph_wins,
                "ties": ties,
                "email_agent_win_rate": email_wins / len(results),
                "langgraph_agent_win_rate": langgraph_wins / len(results),
                "tie_rate": ties / len(results)
            }
        }
    
    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report with tool usage analysis."""
        total_tests = len(results)
        
        # Calculate averages
        avg_similarity = sum(r.response_similarity for r in results) / total_tests
        avg_email_score = sum(r.email_agent_score for r in results) / total_tests
        avg_langgraph_score = sum(r.langgraph_agent_score for r in results) / total_tests
        avg_email_time = sum(r.email_agent_response.processing_time for r in results) / total_tests
        avg_langgraph_time = sum(r.langgraph_agent_response.processing_time for r in results) / total_tests
        
        # Tool usage statistics
        email_tool_usage_count = sum(1 for r in results if r.email_agent_response.tool_usage.tool_used)
        langgraph_tool_usage_count = sum(1 for r in results if r.langgraph_agent_response.tool_usage.tool_used)
        email_correct_tool_usage = sum(1 for r in results if r.email_agent_response.tool_usage.tool_usage_correct)
        langgraph_correct_tool_usage = sum(1 for r in results if r.langgraph_agent_response.tool_usage.tool_usage_correct)
        
        # Calculate wins
        email_wins = sum(1 for r in results if r.email_agent_score > r.langgraph_agent_score)
        langgraph_wins = sum(1 for r in results if r.langgraph_agent_score > r.email_agent_score)
        ties = total_tests - email_wins - langgraph_wins
        
        # Analyze consistency
        consistency_analysis = self._analyze_consistency(results)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_runs": total_tests,
            "test_case": {
                "id": results[0].test_case.id,
                "category": results[0].test_case.category,
                "description": results[0].test_case.description,
                "should_retrieve_info": results[0].test_case.should_retrieve_info
            },
            "overall_metrics": {
                "avg_response_similarity": avg_similarity,
                "avg_email_agent_score": avg_email_score,
                "avg_langgraph_agent_score": avg_langgraph_score,
                "avg_email_agent_time": avg_email_time,
                "avg_langgraph_agent_time": avg_langgraph_time
            },
            "tool_usage_analysis": {
                "email_agent_tool_usage_count": email_tool_usage_count,
                "langgraph_agent_tool_usage_count": langgraph_tool_usage_count,
                "email_agent_correct_tool_usage": email_correct_tool_usage,
                "langgraph_agent_correct_tool_usage": langgraph_correct_tool_usage,
                "email_agent_tool_usage_rate": email_tool_usage_count / total_tests,
                "langgraph_agent_tool_usage_rate": langgraph_tool_usage_count / total_tests,
                "email_agent_correct_tool_usage_rate": email_correct_tool_usage / total_tests,
                "langgraph_agent_correct_tool_usage_rate": langgraph_correct_tool_usage / total_tests
            },
            "performance_comparison": {
                "email_agent_wins": email_wins,
                "langgraph_agent_wins": langgraph_wins,
                "ties": ties,
                "winner": "Email Agent" if email_wins > langgraph_wins else "LangGraph Agent" if langgraph_wins > email_wins else "Tie"
            },
            "consistency_analysis": consistency_analysis,
            "detailed_results": [
                {
                    "run_number": i + 1,
                    "email_agent_response": r.email_agent_response.response,
                    "langgraph_agent_response": r.langgraph_agent_response.response,
                    "email_agent_score": r.email_agent_score,
                    "langgraph_agent_score": r.langgraph_agent_score,
                    "response_similarity": r.response_similarity,
                    "email_agent_time": r.email_agent_response.processing_time,
                    "langgraph_agent_time": r.langgraph_agent_response.processing_time,
                    "tool_usage_comparison": r.tool_usage_comparison
                }
                for i, r in enumerate(results)
            ]
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save the evaluation report to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tool_usage_evaluation_report_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nEvaluation report saved to: {filepath}")
        return filepath
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a summary of the evaluation results with tool usage focus."""
        print("\n" + "="*80)
        print("TOOL USAGE EVALUATION SUMMARY (10 RUNS)")
        print("="*80)
        
        metrics = report["overall_metrics"]
        tool_analysis = report["tool_usage_analysis"]
        comparison = report["performance_comparison"]
        consistency = report.get("consistency_analysis", {})
        
        print(f"Total Runs: {report['total_runs']}")
        print(f"Test Case: {report['test_case']['id']} - {report['test_case']['description']}")
        print(f"Timestamp: {report['timestamp']}")
        print()
        
        print("OVERALL METRICS (Averages):")
        print(f"  Average Response Similarity: {metrics['avg_response_similarity']:.3f}")
        print(f"  Email Agent Average Score: {metrics['avg_email_agent_score']:.3f}")
        print(f"  LangGraph Agent Average Score: {metrics['avg_langgraph_agent_score']:.3f}")
        print(f"  Email Agent Average Time: {metrics['avg_email_agent_time']:.2f}s")
        print(f"  LangGraph Agent Average Time: {metrics['avg_langgraph_agent_time']:.2f}s")
        print()
        
        print("TOOL USAGE ANALYSIS:")
        print(f"  Email Agent Tool Usage: {tool_analysis['email_agent_tool_usage_count']}/{report['total_runs']} ({tool_analysis['email_agent_tool_usage_rate']:.1%})")
        print(f"  LangGraph Agent Tool Usage: {tool_analysis['langgraph_agent_tool_usage_count']}/{report['total_runs']} ({tool_analysis['langgraph_agent_tool_usage_rate']:.1%})")
        print(f"  Email Agent Correct Tool Usage: {tool_analysis['email_agent_correct_tool_usage']}/{report['total_runs']} ({tool_analysis['email_agent_correct_tool_usage_rate']:.1%})")
        print(f"  LangGraph Agent Correct Tool Usage: {tool_analysis['langgraph_agent_correct_tool_usage']}/{report['total_runs']} ({tool_analysis['langgraph_agent_correct_tool_usage_rate']:.1%})")
        print()
        
        print("PERFORMANCE COMPARISON:")
        print(f"  Email Agent Wins: {comparison['email_agent_wins']}")
        print(f"  LangGraph Agent Wins: {comparison['langgraph_agent_wins']}")
        print(f"  Ties: {comparison['ties']}")
        print(f"  Overall Winner: {comparison['winner']}")
        print()
        
        if consistency:
            print("CONSISTENCY ANALYSIS:")
            print("  Tool Usage Consistency:")
            print(f"    Email Agent Tool Usage Rate: {consistency['tool_usage_consistency']['email_agent_tool_usage_rate']:.1%}")
            print(f"    LangGraph Agent Tool Usage Rate: {consistency['tool_usage_consistency']['langgraph_agent_tool_usage_rate']:.1%}")
            print(f"    Email Agent Correct Tool Usage Rate: {consistency['tool_usage_consistency']['email_agent_correct_tool_usage_rate']:.1%}")
            print(f"    LangGraph Agent Correct Tool Usage Rate: {consistency['tool_usage_consistency']['langgraph_agent_correct_tool_usage_rate']:.1%}")
            print()
            
            print("  Score Consistency (Standard Deviation):")
            print(f"    Email Agent Score Std: {consistency['score_consistency']['email_agent_score_std']:.3f}")
            print(f"    LangGraph Agent Score Std: {consistency['score_consistency']['langgraph_agent_score_std']:.3f}")
            print(f"    Email Agent Score Range: {consistency['score_consistency']['email_agent_score_range']}")
            print(f"    LangGraph Agent Score Range: {consistency['score_consistency']['langgraph_agent_score_range']}")
            print()
            
            print("  Time Consistency (Standard Deviation):")
            print(f"    Email Agent Time Std: {consistency['time_consistency']['email_agent_time_std']:.3f}s")
            print(f"    LangGraph Agent Time Std: {consistency['time_consistency']['langgraph_agent_time_std']:.3f}s")
            print(f"    Email Agent Time Range: {consistency['time_consistency']['email_agent_time_range']}")
            print(f"    LangGraph Agent Time Range: {consistency['time_consistency']['langgraph_agent_time_range']}")
            print()
            
            print("  Winner Consistency:")
            print(f"    Email Agent Win Rate: {consistency['winner_consistency']['email_agent_win_rate']:.1%}")
            print(f"    LangGraph Agent Win Rate: {consistency['winner_consistency']['langgraph_agent_win_rate']:.1%}")
            print(f"    Tie Rate: {consistency['winner_consistency']['tie_rate']:.1%}")
            print()

def main():
    """Main function to run the evaluation."""
    print("Email Agent Tool Usage Evaluation Tool")
    print("Comparing EmailRAGAgent vs LangGraphEmailAgent")
    print("Focus: Tool Usage Analysis")
    print("="*80)
    
    # Print configuration
    print("Configuration:")
    print(f"  Evaluation Timeout: {EVALUATION_CONFIG['timeout']}s")
    print(f"  Max Retries: {EVALUATION_CONFIG['max_retries']}")
    print(f"  Output Directory: {EVALUATION_CONFIG['output_dir']}")
    print()
    
    evaluator = EmailAgentEvaluator()
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Generate and save report
    report = evaluator.generate_report(results)
    evaluator.save_report(report)
    
    # Print summary
    evaluator.print_summary(report)

if __name__ == "__main__":
    main() 