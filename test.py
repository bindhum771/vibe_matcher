
from dotenv import load_dotenv
import os
load_dotenv()
print("OPENAI_API_KEY present?:", bool(os.getenv("OPENAI_API_KEY")))
print("Key (first 6 chars):", (os.getenv("OPENAI_API_KEY") or "")[:6])