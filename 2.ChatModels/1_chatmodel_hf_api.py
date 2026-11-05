import os
from dotenv import load_dotenv
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

result = model.invoke("Who is abdul jobayer?")
print(result.content)