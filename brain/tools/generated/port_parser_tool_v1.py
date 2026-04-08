import psutil
import pyperclip

def port_parser_tool() -> dict:
    """
    List all running applications that use the internet and their current ports.
    
    Returns:
        dict: {"success": bool, "result": list of tuples (app_name, port), "error": str or None}
    """

    try:
        result = []
        for conn in psutil.net_connections(kind='inet'):
            app_name = conn.laddr.host + ':' + str(conn.laddr.port)
            result.append((app_name, conn.status))
        
        return {"success": True, "result": result, "error": None}
    
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}