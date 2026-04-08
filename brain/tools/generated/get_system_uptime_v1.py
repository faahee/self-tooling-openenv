import psutil
from datetime import datetime

def get_system_uptime() -> dict:
    """Check system uptime - how long the computer has been running since last restart"""
    try:
        boot_time = psutil.boot_time()
        current_time = datetime.now()
        uptime = (current_time - datetime.fromtimestamp(boot_time)).total_seconds()

        data = {
            "uptime_days": int(uptime / (60 * 60 * 24)),
            "uptime_hours": int((uptime % (60 * 60 * 24)) / (60 * 60)),
            "uptime_minutes": int(((uptime % (60 * 60)) / 60))
        }

        summary = f"System Uptime: {data['uptime_days']} days, {data['uptime_hours']} hours, {data['uptime_minutes']} minutes"

        return {
            "success": True,
            "result": summary,
            "error": None,
            "data": data,
            "source": "psutil",
            "executed_at": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": str(e),
            "executed_at": datetime.now().isoformat()
        }