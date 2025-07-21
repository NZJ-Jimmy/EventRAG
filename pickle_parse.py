import pickle
import os
with open(".pcache/.pkl", "rb") as f:
    openai_complete_if_cache = pickle.load(f)
    
print(openai_complete_if_cache)