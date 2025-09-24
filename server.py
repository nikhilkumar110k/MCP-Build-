from mcp.server.fastmcp import FastMCP
import os, json
import difflib
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from datasets import load_dataset

storage_path="memory_store"
storage_path = storage_path
file = os.path.join(storage_path, "memory.json")
if os.path.exists(file):
            with open(file, "r") as f:
                data = json.load(f)
else:
       data = {}
dataset = load_dataset("csv", data_files="./dataset/medicine_dataset.csv")["train"]
mcp = FastMCP(name="MemoryMCP")

@mcp.tool(description="Save a key-value pair to memory")
def save_memory(key: str, value: str):
        data[key] = value
        _save()
        return f"Memory saved: {key} -> {value}"


SERPAPI_API_KEY = "f37b84c652cd1ce7854120382d33610d5d37f30a7cb757b47d1de8ddd7fd60f4"

def search_web(query: str) -> str:
    if not SERPAPI_API_KEY:
        return "SEARCH_API_KEY not set in .env"

    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "num": 1
    }

    try:
        search1 = GoogleSearch(params)
        results = search1.get_dict() 
        organic = results.get("organic_results")
        if organic and len(organic) > 0:
            snippet = organic[0].get("snippet")
            if snippet:
                return snippet
        return "No results found from web."
    except Exception as e:
        return f"Could not fetch from web: {e}"
    

@mcp.tool(description="Retrieve a value from memory by key, with fuzzy matching")
def get_memory(key: str):
          if key in data:
             return data[key]
          matches = difflib.get_close_matches(key, data.keys(), n=3, cutoff=0.6)
          if matches:
                best_match = matches[0]
                return f"Did you mean '{best_match}'? -> {data[best_match]}"
            
          result = search_web(key)
          save_memory(key, result)
          return f"Web search result for '{key}' -> {result}"

def _save():
        with open(file, "w") as f:
            json.dump(data, f)

@mcp.tool(description="Fetch medicine information from the dataset")
def get_medicine_info(query: str) -> str:
    dataset = load_dataset("csv", data_files="./dataset/medicine_dataset.csv")
    for item in dataset["train"]:
        if query.lower() in item["Name"].lower() or query.lower() in item["Indication"].lower() or query.lower() in item["Classification"].lower() or query.lower() in item["Category"].lower():
            return (
                f"Name: {item['Name']}, Category: {item['Category']}, "
                f"Dosage Form: {item['Dosage Form']}, Strength: {item['Strength']}, "
                f"Manufacturer: {item['Manufacturer']}, Indication: {item['Indication']}, "
                f"Classification: {item['Classification']}"
            )
    return search_web(query)

@mcp.prompt(description="Generate a medical response based on user input and dataset information")
def medical_response_prompt(user_input: str) -> str:
    from server import get_medicine_info
    dataset_info = get_medicine_info(user_input)
    prompt_text = f"""
User query: {user_input}

Relevant medicine info: {dataset_info}

Instruction: Using the above information, generate a detailed, easy-to-understand medical explanation. 
Do not just repeat the text — explain it clearly, including uses, dosage,and precautions if possible.
"""
    return prompt_text


@mcp.tool(description="Generate a medical response based on user input and dataset information")
def medical_response(user_input: str) -> str:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "google/flan-t5-small" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    prompt_text = medical_response_prompt(user_input)

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    outputs = model.generate(**inputs, max_length=256, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response




if __name__ == "__main__":    
    mcp.run(transport="streamable-http",mount_path="/mcp")