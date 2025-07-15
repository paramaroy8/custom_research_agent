"""
In case of "Project limit exceeded", it uses OpenAI based evaluation.
"""

import openai
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from pydantic import BaseModel
from datetime import datetime
import json
import os
from dotenv import load_dotenv

from typing import Optional

# Correct Judgeval tracer imports
from judgeval.tracer import Tracer, wrap
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer

# Load API keys from .env file
load_dotenv()

# Initialize Judgeval tracer (correct syntax)
judgment = Tracer(project_name="research_agent")

# Initialize OpenAI client with tracer wrapper
client = wrap(openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY')))  # This tracks all LLM calls

# Define Pydantic model for parsing response
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    steps: int

# Create LLM output parser
from langchain_core.output_parsers import PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Step 1: Initialize Wikipedia client
import wikipedia
wiki_client = wikipedia

# Step 2: Create the Wikipedia API wrapper
api_wrapper = WikipediaAPIWrapper(
    wiki_client=wiki_client,
    top_k_results=5,
    doc_content_chars_max=1000,
    lang='en'
)

# Step 3: Initialize the WikipediaQueryRun tool
wiki_tool = WikipediaQueryRun(
    api_wrapper=api_wrapper,
    name="wikipedia",
    description="A wrapper around Wikipedia for answering queries",
    verbose=True
)

# Step 4: Set up the DuckDuckGo search tool
search = DuckDuckGoSearchRun()

# Step 5: Create the web search tool for the agent
search_tool = Tool(
    name="web_search",
    func=search.run,
    description="Search web for information",
)

# Function to interact with the wrapped OpenAI client
@judgment.observe(span_type="function")
def openai_chat_completion(messages, model="gpt-4"):
    """Interact with OpenAI API directly to perform chat completion with tracing."""
    try:
        # Ensure the 'model' and 'messages' parameters are passed
        if not messages:
            raise ValueError("The 'messages' parameter cannot be empty.")

        # Call OpenAI Chat Completions API with the wrapped client (automatically traced)
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        # Return the content of the response
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None

# Web search tool with tracing
@judgment.observe(span_type="tool")
def run_web_search(query: str) -> str:
    """Run web search with tracing."""
    try:
        result = search_tool.run(query)
        return result
    except Exception as e:
        print(f"Error in web search: {e}")
        return f"Web search failed: {e}"

# Wikipedia tool with tracing
@judgment.observe(span_type="tool") 
def run_wikipedia_search(query: str) -> str:
    """Run Wikipedia search with tracing."""
    try:
        result = wiki_tool.run(query)
        return result
    except Exception as e:
        print(f"Error in Wikipedia search: {e}")
        return f"Wikipedia search failed: {e}"

# Custom agent function with tracing
@judgment.observe(span_type="function")
def run_custom_agent(user_query: str, max_steps: int):
    """Main function for running the custom agent with integrated tools and OpenAI API."""
    
    # Define initial chat history and other tracking variables
    chat_history = []
    tools_used = []
    steps_taken = 0
    
    # Add system message
    chat_history.append({"role": "system", "content": "You are a research assistant."})
    # Add user query
    chat_history.append({"role": "user", "content": user_query})

    while steps_taken < max_steps:
        # Use search tool with tracing
        search_results = run_web_search(user_query)
        tools_used.append("search_tool")
        chat_history.append({"role": "system", "content": f"Web search results: {search_results}"})
        
        # Use wiki tool with tracing
        wiki_results = run_wikipedia_search(user_query)
        tools_used.append("wiki_tool")
        chat_history.append({"role": "system", "content": f"Wikipedia results: {wiki_results}"})
        
        # Send results to OpenAI API for processing (automatically traced)
        openai_response = openai_chat_completion(chat_history, model="gpt-4")
        
        # Handle case where OpenAI API call failed
        if openai_response is None:
            print("Error: OpenAI API call failed. Stopping agent.")
            break
        
        # Process OpenAI's response and check if it's complete
        if "save to file" in user_query.lower():
            save_to_text(data=openai_response, filename="agent_output.txt")
        
        # Break the loop if response is satisfactory
        if "done" in openai_response.lower():
            break
        
        # Add response to chat history for potential next iteration
        chat_history.append({"role": "assistant", "content": openai_response})
        
        # Update steps taken
        steps_taken += 1
    
    # Return final result, tools used, and steps taken
    return openai_response, tools_used, steps_taken

# Function to save research data to a text file with tracing
@judgment.observe(span_type="tool")
def save_to_text(data, filename="agent_output.txt"):
    """Save research data to file with tracing."""
    try:
        with open(filename, "a") as f:
            f.write("---Research Output---\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")
            
            # If the data is a Pydantic model, convert it to a dictionary
            if hasattr(data, 'model_dump'):
                data = data.model_dump()
            
            if isinstance(data, dict):
                f.write(f"Research on {data.get('topic', 'Unknown')}\n\n")
                f.write(f"Topic: {data.get('topic', 'Unknown')}\n")
                f.write(f"Summary: {data.get('summary', 'No summary provided.')}\n")
                f.write(f"Sources: {data.get('sources', [])}\n")
                f.write(f"Tools Used: {data.get('tools_used', [])}\n")
                f.write(f"Steps Taken: {data.get('steps', 'Unknown')}\n")
            else:
                f.write(f"Data: {data}\n")
            
            f.write("\n" + "="*50 + "\n\n")

        print(f"\t\tDone writing to file: {filename}\n")
        return f"Successfully saved to {filename}"
    except Exception as e:
        print(f"Error saving to file: {e}")
        return f"Failed to save to file: {e}"

# Evaluate the research output using Judgeval with tracing
@judgment.observe(span_type="function")
def evaluate_research_output(user_query: str, final_output: Optional[str], retrieval_context: Optional[list] = None):
    """Evaluate the research output using Judgeval with tracing"""
    
    # Initialize Judgeval client
    client = JudgmentClient()
    
    # Create example for evaluation
    example = Example(
        input=user_query,
        actual_output=str(final_output),
        retrieval_context=retrieval_context or ["Research conducted using web search and Wikipedia tools"]
    )
    
    # Create scorer
    scorer = FaithfulnessScorer(threshold=0.5)
    
    try:
        # Run evaluation
        evaluation_result = client.assert_test(
            examples=[example],
            scorers=[scorer],
            model="gpt-4o-mini",
        )
        print(f"\n--- EVALUATION RESULTS ---")
        print(f"Evaluation completed successfully")
        return evaluation_result
    except Exception as e:
        print(f"\n--- EVALUATION ERROR ---")
        print(f"Evaluation failed with error: {e}")
        return None

# Alternative evaluation using OpenAI with tracing
@judgment.observe(span_type="function")
def evaluate_research_output_openai(user_query: str, final_output: Optional[str], tools_used: list):
    """Evaluate research output using OpenAI directly with tracing"""
    
    evaluation_prompt = f"""
    You are an expert evaluator of research outputs. Please evaluate the following research result:

    Original Query: {user_query}
    Research Output: {final_output}
    Tools Used: {', '.join(tools_used)}

    Please evaluate this research output on the following criteria and provide scores (0-10):

    1. RELEVANCE: How well does the output answer the original query?
    2. ACCURACY: How factually correct is the information provided?
    3. COMPLETENESS: How thorough is the research coverage?
    4. CLARITY: How clear and well-structured is the output?
    5. SOURCE QUALITY: How reliable are the sources used?

    Provide your evaluation in the following JSON format:
    {{
        "relevance_score": 0-10,
        "accuracy_score": 0-10,
        "completeness_score": 0-10,
        "clarity_score": 0-10,
        "source_quality_score": 0-10,
        "overall_score": 0-10,
        "feedback": "Brief explanation of strengths and weaknesses",
        "suggestions": "Suggestions for improvement"
    }}
    """
    
    try:
        evaluation_messages = [
            {"role": "system", "content": "You are an expert research evaluator."},
            {"role": "user", "content": evaluation_prompt}
        ]
        
        # Use the traced OpenAI function
        evaluation_response = openai_chat_completion(evaluation_messages, model="gpt-4")
        
        if evaluation_response:
            print(f"\n--- EVALUATION RESULTS ---")
            print(evaluation_response)
            
            # Try to parse JSON if possible
            try:
                import json
                eval_data = json.loads(evaluation_response)
                print(f"\nOverall Score: {eval_data.get('overall_score', 'N/A')}/10")
                print(f"Feedback: {eval_data.get('feedback', 'N/A')}")
            except json.JSONDecodeError:
                print("Evaluation completed (JSON parsing failed, but response received)")
            
            return evaluation_response
        else:
            print("Evaluation failed - OpenAI API error")
            return None
            
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None

# Main function to run the research agent with full tracing and evaluation
@judgment.observe(span_type="function")
def run_research_agent():
    """Main function to run the research agent with full tracing and evaluation"""
    
    # Take user's query as input
    user_query = input('Hi there! What do you want to learn about today?\t')

    # Ask the user to decide how many steps they want the agent to take for a given query
    max_steps = int(input("Maximum steps for this task?\t"))

    # Ask if user wants evaluation
    run_eval = input("Do you want to run evaluation? (y/n)\t").lower().strip() == 'y'

    print(f"\n--- STARTING RESEARCH WITH TRACING ---")
    print(f"Query: {user_query}")
    print(f"Max steps: {max_steps}")
    print(f"Evaluation: {'Enabled' if run_eval else 'Disabled'}")

    # Execute custom agent (automatically traced)
    final_output, tools_used, steps_taken = run_custom_agent(user_query, max_steps)

    print(f"Final Output: {final_output}")


    # Run evaluation if requested
    if run_eval:
        # Try Judgeval evaluation first
        retrieval_context = tools_used
        evaluation_result = evaluate_research_output(user_query, final_output, retrieval_context)
        
        # If Judgeval evaluation fails (due to quota), use OpenAI evaluation
        if evaluation_result is None:
            print("Falling back to OpenAI-based evaluation...")
            evaluate_research_output_openai(user_query, final_output, tools_used)

    print("\n\n\t\t\t\t\t-----Task Complete-----\n")
        
    return final_output

# Run the main function
if __name__ == "__main__":
    result = run_research_agent()