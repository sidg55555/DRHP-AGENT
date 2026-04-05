import google.generativeai as genai

# 🔹 Set API key manually
genai.configure(api_key="AIzaSyBM47nSazA3KgJdGwsIzzKk93x56ARt7Vc")

# 🔹 Now this will work
from google.generativeai import list_models

for m in list_models():
    print(m.name, m.supported_generation_methods)