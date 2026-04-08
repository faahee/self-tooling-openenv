import psutil
from datetime import datetime, timedelta

def get_last_boot_time() -> dict:
    """
    Returns the last boot time of the system.
    
    Parameters:
    None
    
    Returns:
    dict: {"success": bool, "result": str, "error": str or None}
    """

    try:
        # Get current time
        current_time = datetime.now()
        
        # Get last boot time
        last_boot_time = psutil.boot_time()
        
        # Calculate uptime
        uptime = timedelta(seconds=last_boot_time)
        
        # Convert uptime to string format
        formatted_uptime = str(uptime).split('.')[0]
        
        # Return result
        return {"success": True, "result": f"Last boot time: {formatted_uptime}", "error": None}
    
    except Exception as e:
        # Handle exceptions
        return {"success": False, "result": "", "error": str(e)}