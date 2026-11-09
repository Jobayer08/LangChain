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

chat_history = []

while True:
    user_input=input("You: ")
    chat_history.append(user_input)
    if user_input=="exit":
        break
    result=model.invoke(chat_history)
    chat_history.append(result.content)
    print("Bot: ", result.content)

print(chat_history)    