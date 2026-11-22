from typing import List, Dict, Any
from langchain_core.tools import BaseTool

def convert_langchain_tool_to_gemini(tool: BaseTool) -> Dict[str, Any]:
    """Convert a LangChain BaseTool into a Gemini API FunctionDeclaration dictionary."""
    if not tool.args_schema:
        # If no arguments are needed
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
        
    schema = tool.args_schema.model_json_schema()
    
    properties = schema.get("properties", {})
    clean_properties = {}
    
    for k, v in properties.items():
        clean_prop = dict(v)
        # Remove keys that might confuse the Gemini Live API schema validator
        clean_prop.pop("title", None)
        clean_prop.pop("default", None)
        clean_properties[k] = clean_prop
            
    parameters = {
        "type": "object",
        "properties": clean_properties,
    }
    if "required" in schema:
        parameters["required"] = schema["required"]
        
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": parameters
    }

def get_gemini_tools_from_langchain(tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """Convert a list of LangChain tools to the format expected by LiveConnectConfig."""
    declarations = [convert_langchain_tool_to_gemini(t) for t in tools]
    return [{"function_declarations": declarations}]
