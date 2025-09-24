from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio
import traceback

server_params = StdioServerParameters(
    command="npx",
    args=["mcp-remote","http://127.0.0.1:8000/mcp", "--allow-http"],
)

async def run():
    try:
        print("Starting stdio_client...")
        async with stdio_client(server_params) as (read, write):
            print("Client connected, creating session...")
            async with ClientSession(read, write) as session:

                print("Initializing session...")
                await session.initialize()


                print("Listing tools...")
                tools = await session.list_tools()
                print("Available tools:", tools)

                print("Calling tool...")
                result = await session.call_tool("get_memory", arguments={"key": "general tablet"})
                result3= await session.call_tool("save_memory",arguments={"key":"user","value":"aranav"})
                print("Tool result:", result)
                print("Save result:", result3)

                print("Listing prompts...")
                prompts = await session.list_prompts()
                print("Available prompts templates:", prompts)

                print("Prompt tool...")
                result = await session.get_prompt("medical_response", arguments={"user_input": "Amoxistatin"})
                print("Prompt result:", result)



    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run())
