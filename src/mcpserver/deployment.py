from mcp.server.fastmcp import FastMCP

mcp= FastMCP("BTM MCP Project")

@mcp.tool(description="Fetch data from the source")
def fetch_data():
    """
    Fetch data from the source for BTM dataset fetching.
    """
    return ""


@mcp.resource(description="serching the dataset resources")
def search_dataset():
    """
    search the datset for ther specific data for the medicine 
    """

    
    return ""