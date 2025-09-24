from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import traceback
import re
import json

server_params = StdioServerParameters(
    command="npx",
    args=["mcp-remote", "http://127.0.0.1:8000/mcp", "--allow-http"],
)

async def run():
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List available tools
                tools = await session.list_tools()
                tool_context = "Available tools:\n"
                tool_names = []

                for t in tools:
                    props = ''
                    if hasattr(t, 'inputSchema') and t.inputSchema and 'properties' in t.inputSchema:
                        props = ', '.join(t.inputSchema['properties'].keys())
                    tool_context += f"- {t.name}({props}): {t.description}\n"
                    tool_names.append(t.name)

                # LLM prompt
                user_query = "Tell me about Dolocillin"
                llm_input = f"""You are an AI assistant.
{tool_context}
User query: {user_query}
Decide if you need to call a tool. 
If yes, reply exactly in the format: CALL <tool_name> WITH <JSON arguments>
Otherwise, reply directly with the answer."""

                # Ask the LLM
                result_text = await session.get_prompt(
                    "medical_response", arguments={"user_input": llm_input}
                )
                print("LLM initial output:", result_text)

                # Check if LLM wants to call a tool
                call_pattern = re.search(r"CALL (\w+) WITH (.+)", result_text, re.IGNORECASE)
                if call_pattern:
                    tool_to_call = call_pattern.group(1)
                    tool_args_text = call_pattern.group(2)

                    try:
                        tool_args = json.loads(tool_args_text)
                    except json.JSONDecodeError:
                        tool_args = {}

                    if tool_to_call in tool_names:
                        tool_result = await session.call_tool(tool_to_call, arguments=tool_args)
                        print(f"Tool '{tool_to_call}' output:", tool_result)

                        # Feed tool result back to LLM
                        followup_prompt = f"""User query: {user_query}
Tool called: {tool_to_call}
Tool output: {tool_result}
Provide a final detailed answer using the tool information:"""

                        final_response = await session.get_prompt(
                            "medical_response", arguments={"user_input": followup_prompt}
                        )
                        print("Final LLM response:", final_response)
                    else:
                        print(f"Tool '{tool_to_call}' not found.")
                else:
                    print("Final LLM response (no tool call):", result_text)

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run())
