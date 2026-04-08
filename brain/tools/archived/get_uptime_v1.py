import psutil
from datetime import timedelta, datetime
from pathlib import Path

def get_uptime() -> dict:
    """
    how long has my computer been running since last restart
    """
    try:
        # Get the current uptime using psutil
        boot_time = psutil.boot_time()
        
        if not boot_time:
            return {"success": False, "result": None, "error": "System is not running"}
        
        # Calculate the difference between the current uptime and the previous uptime to get the duration since last restart
        current_uptime = timedelta(seconds=psutil.cpu_times().system.boot_time)
        uptime_diff = current_uptime - boot_time
        
        if not uptime_diff:
            return {"success": False, "result": None, "error": "Insufficient permissions to access system information. Please run the tool as administrator."}
        
        # Format the duration into a human-readable format (e.g., days, hours, minutes)
        days, seconds = divmod(uptime_diff.total_seconds(), 86400)
        hours, seconds = divmod(seconds, 3600)
        mins, secs = divmod(seconds, 60)
        
        # Return the formatted duration
        return {
            "success": True,
            "result": f"{int(days)} days, {int(hours)} hours, {int(mins)} minutes",
            "data": str(uptime_diff),
            "summary": "Duration since last restart",
            "source": "psutil",
            "executed_at": datetime.now()
        }
    
    except ImportError as e:
        if 'psutil' in str(e):
            return {"success": False, "result": None, "error": "psutil not installed. Please install using pip install psutil"}
        elif 'datetime' in str(e):
            return {"success": False, "result": None, "error": "datetime not installed. Please install using pip install datetime"}
    
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": str(e),
            "error_type": type(e).__name__,
            "attempted": "psutil",
            "suggestion": "Please check the system's current user and ensure that the tool has sufficient permissions to access system information.",
            "executed_at": datetime.now()
        }