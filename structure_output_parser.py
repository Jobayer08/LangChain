import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

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

schema=[
    ResponseSchema(name="fact 1",description='fact 1 about the topic'),
    ResponseSchema(name="fact 2",description='fact 2 about the topic'),
    ResponseSchema(name="fact 3",description='fact 3 about the topic') , 
]
parser=StructuredOutputParser.from_response_schemas(schema)
template=PromptTemplate(
    template='Give me 3 facts about {topic}\n{format_instructions}',
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt=template.invoke({"topic":"Python programming language"})

result=model.invoke(prompt)

f_r=parser.parse(result.content)
print(f_r)