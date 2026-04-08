import requests
from pathlib import Path
from typing import Optional

def view_image_online(image_url: str) -> dict:
    """
    View an image online.

    Args:
        image_url (str): The URL of the image to view.

    Returns:
        dict: A dictionary containing a boolean indicating success, the result of the operation, and any error message.
    """

    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            return {"success": True, "result": Path(response.url), "error": None}
        else:
            return {"success": False, "result": None, "error": f"Failed to retrieve image. Status code: {response.status_code}"}

    except requests.exceptions.RequestException as e:
        return {"success": False, "result": None, "error": str(e)}

    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}