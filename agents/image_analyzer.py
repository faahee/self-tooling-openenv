"""Image Analyzer — Local vision model interface using Ollama."""
from __future__ import annotations

import base64
import io
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import httpx
import pyautogui
import pyperclip
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """Analyzes images using local vision models on the RTX 3050.

    Default: moondream (1.8GB VRAM). Deep: LLaVA Q4 for complex analysis.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the image analyzer.

        Args:
            config: Full JARVIS configuration dictionary.
        """
        self.fast_model: str = config["vision"]["fast_model"]
        self.deep_model: str = config["vision"]["deep_model"]
        self.base_url: str = config["llm"]["base_url"]
        self.screenshot_path = Path(config["vision"]["screenshot_path"])
        self.screenshot_path.mkdir(parents=True, exist_ok=True)
        self.max_image_size_mb: int = config["vision"]["max_image_size_mb"]
        self.last_screenshot: str | None = None  # Track last screenshot for follow-ups

    @staticmethod
    def build_prompt(user_input: str) -> str:
        """Convert a vague user question into a detailed vision-model prompt."""
        low = user_input.lower()
        # Screen overview
        if any(k in low for k in ("see my screen", "what's on my screen",
                                   "look at my screen", "what do you see")):
            return ("Describe this screenshot in detail. Identify the exact application "
                    "or software visible (name, version if shown), any open tabs, "
                    "file names, URLs, error messages, and the main content on screen.")
        # Software / app identification
        if any(k in low for k in ("which software", "which app", "what app",
                                   "what software", "what program", "identify")):
            return ("Identify the exact software or application shown in this screenshot. "
                    "Provide the application name, any visible version info, window title, "
                    "and describe what the user is doing in it.")
        # Error identification
        if any(k in low for k in ("error", "bug", "issue", "problem", "wrong")):
            return ("Look at this screenshot carefully. Identify any error messages, "
                    "warnings, or problems visible. Read the exact error text and "
                    "suggest what might be wrong.")
        # Text reading
        if any(k in low for k in ("read", "text", "what does it say", "ocr")):
            return ("Read and transcribe all visible text in this screenshot. "
                    "Include menu items, labels, code, and any other readable content.")
        # Default: pass through with enhancement
        return f"Analyze this screenshot carefully and answer: {user_input}"

    async def analyze_file(
        self, image_path: str, question: str, deep: bool = False
    ) -> str:
        """Analyze an image file with a vision model.

        Args:
            image_path: Path to the image file.
            question: Question to ask about the image.
            deep: If True, use the deep (LLaVA) model.

        Returns:
            Analysis text string.
        """
        p = Path(image_path)
        if not p.exists():
            return f"Image not found: {image_path}"

        image_b64 = self._encode_image(str(p))
        model = self.deep_model if deep else self.fast_model
        return await self._call_vision_model(model, image_b64, question)

    async def analyze_screenshot(
        self, question: str, region: tuple | None = None
    ) -> str:
        """Capture the screen and analyze it.

        Args:
            question: Question to ask about the screenshot.
            region: Optional (left, top, width, height) tuple to capture.

        Returns:
            Analysis text string.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.screenshot_path / f"screenshot_{ts}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        img = pyautogui.screenshot(region=region)
        img.save(str(save_path))
        logger.info("Screenshot saved: %s", save_path)
        self.last_screenshot = str(save_path)

        return await self.analyze_file(str(save_path), question)

    async def analyze_clipboard_image(self, question: str) -> str:
        """Analyze an image from the clipboard.

        Args:
            question: Question to ask about the image.

        Returns:
            Analysis text or "No image found in clipboard".
        """
        try:
            img = Image.grabclipboard()
            if img is None:
                return "No image found in clipboard."

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.screenshot_path / f"clipboard_{ts}.png"
            img.save(str(save_path))
            return await self.analyze_file(str(save_path), question)
        except Exception as exc:
            logger.error("Clipboard image error: %s", exc)
            return f"Could not analyze clipboard image: {exc}"

    async def _call_vision_model(
        self, model: str, image_b64: str, question: str
    ) -> str:
        """Send a request to Ollama with an image.

        Args:
            model: Ollama model name.
            image_b64: Base64-encoded image string.
            question: Prompt / question about the image.

        Returns:
            Model response text.
        """
        payload = {
            "model": model,
            "prompt": question,
            "images": [image_b64],
            "stream": False,
            "options": {"num_predict": 500, "num_gpu": 99},
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate", json=payload
                )
                resp.raise_for_status()
                return resp.json().get("response", "")
        except httpx.ConnectError:
            return "Ollama is not running. Start it with: ollama serve"
        except httpx.TimeoutException:
            return "Vision model timed out."
        except Exception as exc:
            logger.error("Vision model error: %s", exc)
            return f"Could not analyze image: {exc}"

    def _encode_image(self, path: str) -> str:
        """Read, resize if needed, and base64-encode an image.

        Args:
            path: Image file path.

        Returns:
            Base64-encoded string of the image.
        """
        img = Image.open(path)

        # Resize for vision models — moondream works best under 1280px
        max_dim = 1280
        if img.width > max_dim or img.height > max_dim:
            ratio = min(max_dim / img.width, max_dim / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        fmt = "PNG" if img.mode == "RGBA" else "JPEG"
        img.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # ── VISION-GUIDED GUI AUTOMATION ──────────────────────────────────────────

    def _take_grid_screenshot(self, grid_rows: int = 8, grid_cols: int = 10) -> tuple[str, dict]:
        """Capture the screen and overlay a numbered grid for the vision model.

        Each grid cell gets a number label. The vision model can reference
        cells by number, and we map them back to screen coordinates.

        Returns:
            Tuple of (path_to_grid_image, grid_map) where grid_map is
            {cell_number: (center_x, center_y)} in actual screen coordinates.
        """
        img = pyautogui.screenshot()
        screen_w, screen_h = img.width, img.height
        draw = ImageDraw.Draw(img)

        # Calculate cell dimensions
        cell_w = screen_w / grid_cols
        cell_h = screen_h / grid_rows

        # Try to get a readable font
        try:
            font = ImageFont.truetype("arial.ttf", max(14, int(cell_h * 0.2)))
        except Exception:
            font = ImageFont.load_default()

        grid_map: dict[int, tuple[int, int]] = {}
        cell_num = 1

        for row in range(grid_rows):
            for col in range(grid_cols):
                x0 = int(col * cell_w)
                y0 = int(row * cell_h)
                x1 = int((col + 1) * cell_w)
                y1 = int((row + 1) * cell_h)
                cx = (x0 + x1) // 2
                cy = (y0 + y1) // 2

                # Draw grid lines
                draw.rectangle([x0, y0, x1, y1], outline="red", width=1)
                # Draw cell number with background
                label = str(cell_num)
                draw.rectangle([x0 + 2, y0 + 2, x0 + 30, y0 + 20], fill="red")
                draw.text((x0 + 4, y0 + 2), label, fill="white", font=font)

                grid_map[cell_num] = (cx, cy)
                cell_num += 1

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = str(self.screenshot_path / f"grid_{ts}.png")
        img.save(save_path)
        logger.info("Grid screenshot saved: %s (%dx%d grid)", save_path, grid_cols, grid_rows)
        return save_path, grid_map

    async def _ask_vision_for_grid_cell(self, grid_image_path: str, target: str) -> int | None:
        """Ask the vision model which grid cell contains the target element.

        Args:
            grid_image_path: Path to the grid-annotated screenshot.
            target: Description of what to find (e.g., "search bar", "Rifana contact").

        Returns:
            Grid cell number, or None if not found.
        """
        prompt = (
            f"This screenshot has a numbered grid overlay. Each cell has a red number "
            f"in its top-left corner. Look at the image and find: '{target}'. "
            f"Reply with ONLY the grid cell number where '{target}' is located. "
            f"Just the number, nothing else."
        )
        image_b64 = self._encode_image(grid_image_path)
        response = await self._call_vision_model(self.fast_model, image_b64, prompt)
        response = response.strip()

        # Extract the first number from response
        match = re.search(r'\b(\d{1,3})\b', response)
        if match:
            cell = int(match.group(1))
            logger.info("Vision identified '%s' at grid cell %d (raw: '%s')", target, cell, response[:80])
            return cell

        logger.warning("Vision could not locate '%s'. Response: '%s'", target, response[:100])
        return None

    async def vision_click(self, target: str) -> tuple[bool, str]:
        """Use vision to find and click a UI element on screen.

        Takes a grid screenshot, asks moondream where the target is,
        and clicks the center of that grid cell.

        Args:
            target: Natural language description of what to click.

        Returns:
            Tuple of (success, status_message).
        """
        grid_path, grid_map = self._take_grid_screenshot()
        cell = await self._ask_vision_for_grid_cell(grid_path, target)

        if cell is None or cell not in grid_map:
            return False, f"Could not find '{target}' on screen."

        x, y = grid_map[cell]
        pyautogui.click(x, y)
        time.sleep(0.5)
        logger.info("Vision-clicked '%s' at (%d, %d) [cell %d]", target, x, y, cell)
        return True, f"Clicked '{target}' at position ({x}, {y})."

    async def vision_type(self, target: str, text: str) -> tuple[bool, str]:
        """Use vision to find a UI element, click it, and type text.

        Args:
            target: What to click (e.g., "search bar", "message input box").
            text: The text to type after clicking.

        Returns:
            Tuple of (success, status_message).
        """
        success, msg = await self.vision_click(target)
        if not success:
            return False, msg

        time.sleep(0.3)
        # Use clipboard paste for Unicode support
        old_clip = ""
        try:
            old_clip = pyperclip.paste()
        except Exception:
            pass
        pyperclip.copy(text)
        time.sleep(0.1)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.3)
        try:
            pyperclip.copy(old_clip)
        except Exception:
            pass

        return True, f"Typed '{text}' into '{target}'."

    async def vision_gui_agent(self, task: str, max_steps: int = 10) -> str:
        """Autonomous vision-guided GUI automation loop.

        Takes a high-level task, then iterates:
          1. Screenshot with grid
          2. Ask vision model what to do next
          3. Execute the action (click / type / scroll / press key)
          4. Repeat until task is done or max steps reached

        Args:
            task: Natural language description of the full task.
            max_steps: Maximum number of action steps.

        Returns:
            Summary of what was done.
        """
        actions_taken: list[str] = []

        for step in range(1, max_steps + 1):
            # 1. Capture screen with grid
            grid_path, grid_map = self._take_grid_screenshot()
            image_b64 = self._encode_image(grid_path)

            # 2. Ask vision model what action to take
            history = "\n".join(f"  Step {i+1}: {a}" for i, a in enumerate(actions_taken))
            prompt = (
                f"You are a GUI automation assistant. The screenshot has a numbered grid overlay.\n"
                f"Task: {task}\n"
                f"Actions taken so far:\n{history if history else '  (none)'}\n\n"
                f"Look at the current screenshot and decide the NEXT single action to take.\n"
                f"Reply in EXACTLY this format (one line only):\n"
                f"  CLICK <cell_number> - <description of what you're clicking>\n"
                f"  TYPE <cell_number> <text> - <description>\n"
                f"  SCROLL <up/down> - <reason>\n"
                f"  KEY <key_name> - <reason> (e.g., KEY enter, KEY escape, KEY tab)\n"
                f"  DONE - <summary of what was accomplished>\n\n"
                f"Reply with exactly ONE action line."
            )

            response = await self._call_vision_model(self.fast_model, image_b64, prompt)
            response = response.strip().split('\n')[0].strip()  # First line only
            logger.info("Vision GUI step %d: %s", step, response[:100])

            # 3. Parse and execute the action
            resp_upper = response.upper()

            if resp_upper.startswith("DONE"):
                actions_taken.append(f"DONE: {response}")
                break

            elif resp_upper.startswith("CLICK"):
                match = re.search(r'CLICK\s+(\d+)', response, re.IGNORECASE)
                if match:
                    cell = int(match.group(1))
                    if cell in grid_map:
                        x, y = grid_map[cell]
                        pyautogui.click(x, y)
                        time.sleep(0.8)
                        actions_taken.append(f"Clicked cell {cell} at ({x},{y}): {response}")
                    else:
                        actions_taken.append(f"Invalid cell {cell}")
                else:
                    actions_taken.append(f"Could not parse CLICK: {response}")

            elif resp_upper.startswith("TYPE"):
                match = re.search(r'TYPE\s+(\d+)\s+(.+?)(?:\s*-\s*|$)', response, re.IGNORECASE)
                if match:
                    cell = int(match.group(1))
                    text = match.group(2).strip()
                    if cell in grid_map:
                        x, y = grid_map[cell]
                        pyautogui.click(x, y)
                        time.sleep(0.3)
                        old_clip = ""
                        try:
                            old_clip = pyperclip.paste()
                        except Exception:
                            pass
                        pyperclip.copy(text)
                        pyautogui.hotkey('ctrl', 'v')
                        time.sleep(0.5)
                        try:
                            pyperclip.copy(old_clip)
                        except Exception:
                            pass
                        actions_taken.append(f"Typed '{text}' at cell {cell}: {response}")
                    else:
                        actions_taken.append(f"Invalid cell {cell}")
                else:
                    actions_taken.append(f"Could not parse TYPE: {response}")

            elif resp_upper.startswith("SCROLL"):
                if "down" in response.lower():
                    pyautogui.scroll(-3)
                else:
                    pyautogui.scroll(3)
                time.sleep(0.5)
                actions_taken.append(f"Scrolled: {response}")

            elif resp_upper.startswith("KEY"):
                match = re.search(r'KEY\s+(\w+)', response, re.IGNORECASE)
                if match:
                    key = match.group(1).lower()
                    pyautogui.press(key)
                    time.sleep(0.5)
                    actions_taken.append(f"Pressed '{key}': {response}")
                else:
                    actions_taken.append(f"Could not parse KEY: {response}")
            else:
                actions_taken.append(f"Unknown action: {response}")

        summary = f"Vision GUI agent completed in {len(actions_taken)} steps:\n"
        for i, a in enumerate(actions_taken, 1):
            summary += f"  {i}. {a}\n"
        return summary
