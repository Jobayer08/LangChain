from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)
parser=JsonOutputParser()

template=PromptTemplate(
    template="give me the 5 fact about {topic}\n{fictional_instruction}",
    input_variables=["topic"],
    partial_variables={"fictional_instruction":parser.get_format_instructions()}

)

chain=template | model | parser
f_r=chain.invoke({"topic": "black hole"})

print(f_r)