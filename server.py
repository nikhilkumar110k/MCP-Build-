from mcp.server.fastmcp import FastMCP
import os, json
import difflib

storage_path="memory_store"
storage_path = storage_path
file = os.path.join(storage_path, "memory.json")
if os.path.exists(file):
            with open(file, "r") as f:
                data = json.load(f)
else:
       data = {}

mcp = FastMCP(name="MemoryMCP")

@mcp.tool(description="Save a key-value pair to memory")
def save_memory(key: str, value: str):
        data[key] = value
        _save()
        return f"Memory saved: {key} -> {value}"

@mcp.tool(description="Retrieve a value from memory by key, with fuzzy matching")
def get_memory(key: str):
          if key in data:
             return data[key]
          matches = difflib.get_close_matches(key, data.keys(), n=3, cutoff=0.6)
          if matches:
                best_match = matches[0]
                return f"Did you mean '{best_match}'? -> {data[best_match]}"
            
          return f"No memory found similar to: '{key}'"

def _save():
        with open(file, "w") as f:
            json.dump(data, f)


if __name__ == "__main__":    
    mcp.run(transport="streamable-http")