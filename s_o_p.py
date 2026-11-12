from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

templete1=PromptTemplate(
    template="write a detail report on {topic}",
    input_variables=["topic"]
)

templete2=PromptTemplate(
    template="write a 5 line summary on the following text:/n{text}",
    input_variables=["text"]
)

parser=StrOutputParser()

chain= templete1 | model| parser | templete2 | model | parser

result=chain.invoke({"topic":"AI"})

print(result)