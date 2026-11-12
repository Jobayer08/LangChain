from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
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

prompt1=templete1.invoke({"topic":"AI"})

result1=model.invoke(prompt1)

prompt2=templete2.invoke({"text":result1.content})

result2=model.invoke(prompt2)

print(result2.content)