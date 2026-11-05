from langchain_huggingface import HuggingFaceEmbeddings 

embeddings = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text="Bangladesh is a country in South Asia. It is bordered by India to the west, north, and east, Myanmar to the southeast, and the Bay of Bengal to the south. The capital of Bangladesh is Dhaka."

vector=embeddings.embed_query(text)
print(str(vector))