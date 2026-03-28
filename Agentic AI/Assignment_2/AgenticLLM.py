import os
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

@tool
def read_local_knowledge(query: str) -> str:
    filename = "knowledge_base.txt"
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("Foundational Knowledge: AI is rapidly transforming modern industries through automation and predictive analytics.")
    
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

def run_local_research_agent(topic: str):

    llm = ChatOllama(model="llama3.1", temperature=0)

    search = DuckDuckGoSearchResults()
    tools = [search, read_local_knowledge]

    agent = create_react_agent(llm, tools=tools)

    system_instructions = f"""
    You are an autonomous research agent. Your task is to research: '{topic}'
    
    Use your tools to gather comprehensive information. 
    You must use the web search tool for recent data and the local knowledge tool for foundational knowledge.
    
    Once you have gathered enough information, you MUST format your final response EXACTLY according to the structure below:
    
    =========================================
    COVER PAGE
    Title: [Insert Catchy Title Here]
    Topic: {topic}
    Author: Autonomous Research Agent
    =========================================
    
    # [Insert Title]
    
    ## Introduction
    [Write a comprehensive introduction to the topic]
    
    ## Key Findings
    [Extract and bullet point the most critical data and insights]
    
    ## Challenges
    [Detail the current obstacles or limitations]
    
    ## Future Scope
    [Discuss what the future holds for this topic]
    
    ## Conclusion
    [Provide a final summary wrapping up the report]
    """

    print(f"Starting local research on: '{topic}'...\n")
    try:
        response = agent.invoke({"messages": [HumanMessage(content=system_instructions)]})
        return response["messages"][-1].content
    except Exception as e:
        return f"An error occurred during agent execution: {e}"

if __name__ == "__main__":
    user_topic = input("Enter a research topic (e.g., 'Impact of AI in Healthcare'): ")
    
    if not user_topic:
        user_topic = "Impact of AI in Healthcare" 
        
    final_report = run_local_research_agent(user_topic)
    
    print("\n\n" + "="*50 + " FINAL REPORT " + "="*50 + "\n")
    print(final_report)
    
    safe_filename = user_topic.replace(" ", "_").replace("/", "_") + "_local_output.txt"
    with open(safe_filename, "w", encoding="utf-8") as file:
        file.write(final_report)
        
    print(f"\n✅ Report saved as '{safe_filename}'.")