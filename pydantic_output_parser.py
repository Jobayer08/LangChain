import os

from dotenv import load_dotenv

from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
load_dotenv(override=True)

llm = HuggingFaceEndpoint(
    # নিচের যেকোনো একটি repo_id ব্যবহার করুন (বাকি দুটি কমেন্ট করে রাখুন):
    #repo_id="meta-llama/Llama-3.1-8B-Instruct",
    repo_id="openai/gpt-oss-120b",
     #repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    #repo_id="Qwen/Qwen3-32B",
    task="text-generation",
    provider="groq",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

class person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    city: str = Field(description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=person)    

template=PromptTemplate(
    template='generate the name, age, and city of a fictional {place} person\n{format_instructions}',
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt=template.invoke({"place":"Bangladeshi"})
print(prompt)
result=model.invoke(prompt)
f_r=parser.parse(result.content)
print(f_r)