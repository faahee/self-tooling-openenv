import time
from typing import Dict, Any

def timer_tool(interval: float = 10) -> Dict[str, Any]:
    """
    Continuously checks CPU temperature every interval seconds for 1 minute.
    
    Args:
        interval (float): Time in seconds to wait between checks. Defaults to 10.

    Returns:
        Dict[str, Any]: {"success": bool, "result": <value>, "error": None/str}
    """
    try:
        # Initialize GPUtil
        import GPUtil
        
        # Calculate uptime
        from datetime import datetime
        boot_time = GPUtil.getGPUs()[0].bootTime
        uptime = (datetime.now() - datetime.fromtimestamp(boot_time)).total_seconds()
        
        # Check CPU temperature every interval seconds for 1 minute
        result = {}
        start_time = time.time()
        while True:
            try:
                # Get current CPU temperature
                cpu_temp = GPUtil.getGPUs()[0].temperature
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Check if 1 minute has passed
                if elapsed_time >= 60:
                    result["success"] = True
                    result["result"] = f"CPU temperature: {cpu_temp}°C, Uptime: {uptime:.2f}s"
                    break
            
            except Exception as e:
                result["error"] = str(e)
                break
            
            # Wait for interval seconds
            time.sleep(interval)
        
        return {"success": True, "result": result["result"], "error": None}
    
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}