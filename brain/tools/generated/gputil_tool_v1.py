import time
from typing import Dict, Any
import GPUtil

def gputil_tool(interval: float = 10) -> Dict[str, Any]:
    """
    Check CPU temperature every interval seconds for 1 minute.

    Args:
        interval (float): Time interval in seconds. Defaults to 10.

    Returns:
        Dict[str, Any]: {"success": bool, "result": <value>, "error": None/str}
    """

    try:
        # Initialize GPUtil
        GPUtil.init()

        result = {}
        start_time = time.time()
        while True:
            try:
                # Get CPU temperature
                cpu_temp = GPUtil.getGPUs()[0].temperature

                # Calculate uptime
                uptime = int(time.time() - GPUtil.boot_time())

                # Store results in dictionary
                result["cpu_temp"] = cpu_temp
                result["uptime"] = uptime

                # Break loop after 1 minute
                if time.time() - start_time >= 60:
                    break

            except Exception as e:
                result["error"] = str(e)
                break

            # Wait for interval seconds
            time.sleep(interval)

        return {"success": True, "result": result, "error": None}

    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}