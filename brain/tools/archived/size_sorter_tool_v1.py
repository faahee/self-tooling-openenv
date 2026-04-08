import pathlib
import psutil

def size_sorter_tool(root_dir: str = None) -> dict:
    """
    Find the 5 biggest files on the C drive and tell me their sizes.

    Args:
        root_dir (str): The path to the directory to search for files. Defaults to None, which means the current working directory.

    Returns:
        dict: A dictionary containing a boolean indicating success, the result of the operation, and an error message if any.
    """

    try:
        # Get the C drive path
        c_drive = pathlib.Path(root_dir or os.getcwd()).parent

        # List files on the C drive
        files = [f for f in c_drive.iterdir() if f.is_file()]

        # Sort files by size in descending order
        sorted_files = sorted(files, key=lambda x: x.stat().st_size, reverse=True)

        # Get the 5 biggest files
        biggest_files = sorted_files[:5]

        # Return their sizes as a list
        return {"success": True, "result": [f.stat().st_size for f in biggest_files], "error": None}

    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}