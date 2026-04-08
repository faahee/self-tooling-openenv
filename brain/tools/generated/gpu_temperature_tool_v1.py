import GPUtil

def gpu_temperature_tool() -> dict:
    """
    Access system data for GPU temperature.

    Returns:
        dict: {"success": bool, "result": <value>, "error": None/str}
    """

    try:
        gpu = GPUtil.getGPUs()[0]
        result = gpu.temperature
        return {"success": True, "result": result, "error": None}

    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}