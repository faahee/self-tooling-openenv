import pyperclip
from typing import Dict, Any

def pyperclip_agent() -> Dict[str, Any]:
    """
    Returns the last clipboard content.
    
    Parameters:
    None
    
    Returns:
    A dictionary with 'success' (bool), 'result' (str), and optional 'error' (str).
    """

    try:
        result = pyperclip.paste()
        return {"success": True, "result": result, "error": None}
    
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}