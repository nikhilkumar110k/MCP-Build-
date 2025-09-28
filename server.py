from mcp.server.fastmcp import FastMCP
import os, json
import difflib
from serpapi import GoogleSearch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import re


embed_model = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embed_model)
model = AutoModel.from_pretrained(embed_model)

def get_embedding(text: str) -> np.ndarray:
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
        embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

embedding_store = {}


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



def khm_match(query: str, k: int = 3):
    if not embedding_store:
        return None
    
    query_vec = get_embedding(query)
    keys = list(embedding_store.keys())
    X = np.array([embedding_store[k] for k in keys])

    dists = euclidean_distances([query_vec], X)[0]
    dists = np.where(dists == 0, 1e-6, dists)

    idx_sorted = np.argsort(dists)
    topk = dists[idx_sorted[:k]]
    harmonic_mean = k / np.sum(1.0 / topk)

    best_idx = idx_sorted[0]
    return keys[best_idx], data[keys[best_idx]], harmonic_mean



@mcp.tool(description="Save a key-value pair to memory")
def save_memory(key: str, value: str):
    data[key] = value
    embedding_store[key] = get_embedding(key).tolist()  
    _save()
    return f"Memory saved: {key} -> {value}"




SERPAPI_API_KEY = "f37b84c652cd1ce7854120382d33610d5d37f30a7cb757b47d1de8ddd7fd60f4"

def search_web(query: str) -> str:
    if not SERPAPI_API_KEY:
        return "SEARCH_API_KEY not set in .env"

    params = {
        "q": f"{query}",
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "num": 3
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
    
@mcp.tool(description="Suggest alternative medicines based on category, classification, strength, and dosage")
def recommend_alternatives(target_medicine: dict, max_results: int = 3, strength_tolerance: int = 100):
    candidates = []
    try:
        target_strength = int(target_medicine["Strength"].split()[0])
    except:
        target_strength = None

    target_class = target_medicine["Classification"]
    target_cat = target_medicine["Category"]
    target_dosage = target_medicine["Dosage Form"]

    for item in dataset:
        if item["Name"] == target_medicine["Name"]:
            continue

        same_class = item["Classification"] == target_class
        same_cat = item["Category"] == target_cat
        same_dosage = item["Dosage Form"] == target_dosage

        close_strength = False
        if target_strength is not None:
            try:
                item_strength = int(item["Strength"].split()[0])
                close_strength = abs(item_strength - target_strength) <= strength_tolerance
            except:
                pass

        if same_class or same_cat or close_strength or same_dosage:
            candidates.append(item)

    return candidates[:max_results]

@mcp.tool(description="Retrieve a value from memory by key, with KHM fuzzy matching")
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



def parse_medicine_snippet(snippet: str):
    info = {}
    patterns = {
        "Name": r"Name[:\-]?\s*([A-Za-z0-9\s]+)",
        "Category": r"Category[:\-]?\s*([A-Za-z\s]+)",
        "Dosage Form": r"Dosage Form[:\-]?\s*([A-Za-z\s]+)",
        "Strength": r"Strength[:\-]?\s*([\d\s\w]+)",
        "Manufacturer": r"Manufacturer[:\-]?\s*([A-Za-z\s]+)",
        "Classification": r"Classification[:\-]?\s*([A-Za-z\s]+)",
        "Indication": r"Indication[:\-]?\s*([A-Za-z\s]+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, snippet, re.IGNORECASE)
        info[key] = match.group(1).strip() if match else None
    return info


@mcp.tool(description="Fetch medicine information from the dataset or web")
def get_medicine_info(query: str):
    dataset = load_dataset("csv", data_files="./dataset/medicine_dataset.csv")["train"]

    for item in dataset:
        if query.lower() in item["Name"].lower() \
           or query.lower() in item["Indication"].lower() \
           or query.lower() in item["Classification"].lower() \
           or query.lower() in item["Category"].lower():

            med_info = {
                "Name": item["Name"],
                "Category": item["Category"],
                "Dosage Form": item["Dosage Form"],
                "Strength": item["Strength"],
                "Manufacturer": item["Manufacturer"],
                "Classification": item["Classification"],
                "Indication": item["Indication"]
            }

            med_info["Alternatives"] = recommend_alternatives(item)
            return med_info

    snippet = search_web(query)
    med_info = {
        "Name": query,
        "Category": None,
        "Dosage Form": None,
        "Strength": None,
        "Manufacturer": None,
        "Classification": None,
        "Indication": None,
        "WebSnippet": snippet
    }
    return med_info

@mcp.prompt(description="Generate a medical response based on user input and dataset information")
def medical_response_prompt(user_input: str) -> str:
    from server import get_medicine_info
    dataset_info = get_medicine_info(user_input)
    prompt_text = f"""
User query: {user_input}

Relevant medicine info: {dataset_info}

Instruction: Using the above information, generate a detailed, easy-to-understand medical explanation. 
Do not just repeat the text — explain it clearly, including Name, dosage, Category and Strength if possible. also tell alternatives to the medicine if any.
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


