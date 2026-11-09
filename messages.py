from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv(override=True)

llm = HuggingFaceEndpoint(
    
    repo_id="openai/gpt-oss-120b",
     
    task="text-generation",
    provider="groq",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is langchain")
    ]

result=model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)