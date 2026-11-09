from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain}."),
    ('human', "explain {topic} in simple terms.")
])

prompt=chat_template.invoke({'domain': 'math teacher', 'topic': 'calculus'})
print(prompt)