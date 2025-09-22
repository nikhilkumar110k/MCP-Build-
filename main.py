from server import save_memory, get_memory
from mcp.server.fastmcp import FastMCP

mcp=FastMCP(name="TestMCP",tools=[get_memory,save_memory])
def main():
    mcp.call_tool("save_memory", arguments={"key": "my mcp worked!", "value": "woohooo"})
    mcp.call_tool("get_memory", arguments={"key": "my mcp worked!"})


if __name__ == "__main__":
    main()
