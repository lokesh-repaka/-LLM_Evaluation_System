import sys
from pathlib import Path

# Add project root to sys.path so the module can be run directly from inside RAG1/
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from RAG1 import tools as tl
from RAG1.tools import collecting_info
from RAG1.prompts import system_agent_prompt
from RAG1.objects import Answer

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

agent = create_agent(
    model=llm,
    tools=[collecting_info],
    system_prompt=system_agent_prompt,
    response_format=Answer
)

def ask(question) -> dict:
    """ Answer the user's question about the pdf"""

    tl.last_retrieved_context = []

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    structured_response = result["structured_response"]

    return {
        "answer" : structured_response.answer,
        "context": structured_response.context
    } 

if __name__ == "__main__":
    print(ask("what is the diet of Penguins"))