from dotenv import find_dotenv, load_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

load_dotenv(find_dotenv())


# --------------------------------------------------------------
# LLMs: Get predictions from a language model
# --------------------------------------------------------------
question = "Who won the FIFA World Cup in the year 1994? "

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# --------------------------------------------------------------
# LLMs: Get predictions from a language model
# --------------------------------------------------------------
repo_id = "google/flan-t5-xxl"

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1})

llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run(question))
