def get_system_restart_duration() -> dict:
    """
    how long has my computer been running since last restart
    """

    try:
        # Import necessary modules
        import psutil
        from datetime import datetime

        # Get current system time using datetime.now()
        current_time = datetime.now()

        # Get last system restart time from Windows registry or WMI
        try:
            # Attempt to get the uptime in seconds
            uptime_seconds = psutil.boot_time()
            if uptime_seconds is None:
                return {"success": False, "result": None, "error": "Insufficient permissions to access system information"}
            else:
                # Convert uptime from seconds to days, hours, minutes, and seconds
                uptime_days, uptime_hours, uptime_minutes, uptime_seconds = divmod(uptime_seconds, 86400)
                uptime_hours, uptime_minutes, uptime_seconds = divmod(uptime_hours, 3600)
                uptime_minutes, uptime_seconds = divmod(uptime_minutes, 60)

        except (ImportError, psutil.NoSuchProcess):
            return {"success": False, "result": None, "error": "some_module not installed"}

        # Calculate difference between current and last restart times
        time_diff_days = (current_time - datetime.fromtimestamp(uptime_seconds)).days
        time_diff_hours, time_diff_minutes, time_diff_seconds = divmod((current_time - datetime.fromtimestamp(uptime_seconds)).total_seconds() % 86400, 3600)
        time_diff_minutes, time_diff_seconds = divmod(time_diff_minutes, 60)

        # Return the result as a dictionary
        return {"success": True, "result": f"Days: {time_diff_days}, Hours: {time_diff_hours}, Minutes: {time_diff_minutes}, Seconds: {time_diff_seconds}"}

    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}