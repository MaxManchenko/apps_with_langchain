import textwrap
from dotenv import find_dotenv, load_dotenv

from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, ConversationChain
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.agents.load_tools import get_all_tool_names

# Load environment variables
load_dotenv(find_dotenv())


# --------------------------------------------------------------
# LLMs: Get predictions from a language model
# --------------------------------------------------------------
# Prompt templates
product = "Smart Apps using Large Language Models (LLMs)"
template = "What is a good name for a company that makes {product}"
prompt = PromptTemplate(template=template, input_variables=["product"])

# Initialise and run the LLM
repo_id = "google/flan-t5-xxl"  # https://huggingface.co/google/flan-t5-xxl
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.6})
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(product))

# --------------------------------------------------------------
# Memory: Add State to Chains and Agents
# --------------------------------------------------------------
repo_id = "google/flan-t5-xxl"  # https://huggingface.co/google/flan-t5-xxl
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.6})
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")
print(output)

output = conversation.predict(input="Nice to meet you! What's your name?")
print(output)

# --------------------------------------------------------------
# Chains: Combine LLMs and prompts in multy-step workflow
# --------------------------------------------------------------
# Prompt templates
product = "AI chatbots for dental offices"
template = "What is a good name for a company that makes {product}?"
prompt = PromptTemplate(template=template, input_variables=["product"])

# Initialise and run the LLM
repo_id = "google/flan-t5-xxl"  # https://huggingface.co/google/flan-t5-xxl
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.6})

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run(product))

# --------------------------------------------------------------
# Agents: Dynamically call chains based on user input
# --------------------------------------------------------------
# Prompt
prompt = "In what year was Python released and who is original creator?"

# Initialize and run the LLM
repo_id = "google/flan-t5-xxl"  # https://huggingface.co/google/flan-t5-xxl
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.7})

get_all_tool_names()
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# Initialize an agent with the tools, the LLM, and the agent type
agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION
agent_chain = initialize_agent(
    tools,
    llm=llm,
    agent=agent,
    verbose=True,
    handle_parsing_errors=True,
)
print(agent_chain.run(prompt))
