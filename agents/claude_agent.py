"""
Claude (Anthropic) agent — uses the computer-use beta API.

Env vars required:
  ANTHROPIC_API_KEY           (for Anthropic direct)
  AWS_ACCESS_KEY_ID + etc.    (for Bedrock)
  bitcy_api_key (or BLTCY_API_KEY)   (for bltcy.ai Anthropic-compatible proxy)
"""

import base64
import io
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast

from PIL import Image

from .base import BaseAgent

logger = logging.getLogger("agents.claude")

# Computer use version configuration
# - 2025-11-24: Latest Anthropic API version (api.anthropic.com and bltcy default channel)
# - 2025-01-24: Older version (some proxies)
# - 2024-10-22: AWS Bedrock version
COMPUTER_USE_BETA_FLAG = "computer-use-2025-11-24"
PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"
COMPUTER_USE_TYPE = "computer_20251124"
# Models that only accept the per-action computer toolset (one tool per action, e.g. left_click {"coordinate": ...});
# its calls are mapped back to {"action": name, ...} so the rest of the agent is shared.
COMPUTER_TOOLSET_TYPE = "computer_toolset_20260801"
TOOLSET_MODELS = ("claude-opus-5-5",)

# bltcy.ai Anthropic-compatible proxy. The SDK appends `/v1/messages`, so the
# base URL here is the API root (not `/v1`).
BLTCY_BASE_URL = "https://api.bltcy.ai"

# ── System prompts ──────────────────────────────────────────────────────

SYSTEM_PROMPT_UBUNTU = """<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using x86_64 architecture with internet access.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* DO NOT ask users for clarification during task execution. DO NOT stop to request more information from users. Always take action using available tools.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* TASK FEASIBILITY: You can declare a task infeasible at any point during execution. If you determine that a task cannot be completed, output exactly "[INFEASIBLE]" (including the square brackets) anywhere in your response to trigger the fail action.
* The current date is {date}.
* Home directory of this Ubuntu system is '/home/user'.
* If you need a password for sudo, the password of the computer is '{password}'.
</SYSTEM_CAPABILITY>

<COMPUTER_TOOL_USAGE>
* You MUST use the 'computer' tool with explicit parameters:
  - action: the action type (click, type, key, scroll, etc.)
  - coordinate: [x, y] coordinates for click-type actions
* Always provide these parameters in your computer tool calls.
* Do not call the computer tool without proper parameters.
</COMPUTER_TOOL_USAGE>"""

SYSTEM_PROMPT_WINDOWS = """<SYSTEM_CAPABILITY>
* You are utilising a Windows virtual machine using x86_64 architecture with internet access.
* Multiple browsers are installed: Google Chrome, Firefox, Brave, and Opera. Launch them from the taskbar or via command line.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* The current date is {date}.
* Home directory of this Windows system is 'C:\\Users\\user'.
* When you want to open some applications on Windows, please use Double Click on it instead of clicking once.
* If you need a password for sudo, The password of the computer is '{password}'.
</SYSTEM_CAPABILITY>"""


class ClaudeAgent(BaseAgent):
    """
    Agent that uses Anthropic's computer-use beta API.

    Supports providers: anthropic (direct), bedrock, vertex.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        provider: str = "anthropic",
        api_backend: str = "anthropic",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        no_thinking: bool = False,
        use_isp: bool = False,
        only_n_most_recent_images: int = 10,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.provider = provider  # "anthropic", "bedrock", "vertex"
        self.api_backend = api_backend.lower()  # "anthropic" or "bltcy"

        if self.api_backend == "bltcy":
            self.api_key = (
                api_key
                or os.environ.get("bitcy_api_key")
                or os.environ.get("BLTCY_API_KEY")
            )
            self.base_url = base_url or BLTCY_BASE_URL
        else:
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self.base_url = base_url

        self.no_thinking = no_thinking
        self.use_isp = use_isp
        self.only_n_most_recent_images = self.history_images or only_n_most_recent_images

        # Claude operates on 1280x720; actual screen is self.screen_size
        self.resize_factor = (
            self.screen_size[0] / 1280,
            self.screen_size[1] / 720,
        )

        self.messages: list = []

    # ── Public interface ────────────────────────────────────────────────

    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        from anthropic import (
            Anthropic,
            AnthropicBedrock,
            AnthropicVertex,
            APIError,
            APIResponseValidationError,
            APIStatusError,
        )
        from anthropic.types.beta import BetaTextBlockParam

        # Resize screenshot to 1280x720 for Claude
        screenshot_bytes = self._resize_screenshot(obs["screenshot"])

        # Build system prompt
        if self.platform == "windows":
            sys_text = SYSTEM_PROMPT_WINDOWS.format(
                date=datetime.today().strftime("%A, %B %d, %Y"),
                password=self.password,
            )
        else:
            sys_text = SYSTEM_PROMPT_UBUNTU.format(
                date=datetime.today().strftime("%A, %B %d, %Y"),
                password=self.password,
            )

        system = BetaTextBlockParam(type="text", text=sys_text)

        # First message: screenshot + instruction
        if not self.messages:
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_b64,
                            },
                        },
                        {"type": "text", "text": instruction},
                    ],
                }
            )
        else:
            # Add tool results for previous tool_use blocks
            self._add_tool_results(screenshot_bytes)

        # Create client
        client = self._create_client(
            Anthropic, AnthropicBedrock, AnthropicVertex
        )

        # Configure betas
        betas = [COMPUTER_USE_BETA_FLAG]
        if self.use_isp:
            betas.append("interleaved-thinking-2025-05-14")
        # Prompt caching: on for direct Anthropic. Disabled for bltcy because
        # measurement shows the proxy writes cache entries but never reads them
        # back (load-balanced upstream), turning caching into a 1.25x cost
        # multiplier instead of a discount.
        if self.provider == "anthropic" and self.api_backend != "bltcy":
            betas.append(PROMPT_CACHING_BETA_FLAG)
            system["cache_control"] = {"type": "ephemeral"}

        # Filter old images
        if self.only_n_most_recent_images:
            self._filter_old_images(self.only_n_most_recent_images)

        # Tool config
        if self._uses_toolset():
            tools = [{"type": COMPUTER_TOOLSET_TYPE}]  # display size is not configurable; zoom is built in
        else:
            tools = [
                {
                    "name": "computer",
                    "type": COMPUTER_USE_TYPE,
                    "display_width_px": 1280,
                    "display_height_px": 720,
                    "display_number": 0,
                    "enable_zoom": True,
                }
            ]

        # Thinking config
        extra_body, actual_max_tokens = self._thinking_config()

        # Sampling params
        sampling = {}
        if self.temperature is not None:
            sampling["temperature"] = self.temperature
        if self.top_p is not None:
            sampling["top_p"] = self.top_p

        # API call with retries
        response = None
        for attempt in range(self.api_max_retries):
            try:
                response = client.beta.messages.create(
                    max_tokens=actual_max_tokens,
                    messages=self.messages,
                    model=self.model,
                    system=[system],
                    tools=tools,
                    betas=betas,
                    extra_body=extra_body,
                    # An explicit timeout lets the SDK accept large max_tokens without streaming.
                    timeout=1200,
                    **sampling,
                )
                break
            except (APIError, APIStatusError, APIResponseValidationError) as e:
                error_msg = str(e)
                logger.warning(f"API error (attempt {attempt + 1}): {error_msg}")
                if attempt < self.api_max_retries - 1:
                    time.sleep(self.api_retry_interval)
                else:
                    logger.error(f"All {self.api_max_retries} attempts failed")
                    return "", ["FAIL"]

        if response is None:
            return "", ["FAIL"]

        # Parse response into message params
        response_params = self._response_to_params(response)

        # Store in history
        self.messages.append({"role": "assistant", "content": response_params})

        # Extract reasoning + actions
        reasoning, actions = self._parse_response(response_params, response)

        # Store raw response for logging
        try:
            self.last_raw_response = response.model_dump()
        except Exception:
            self.last_raw_response = {"content": response_params}

        return reasoning, actions

    def reset(self) -> None:
        self.messages = []

    def _uses_toolset(self) -> bool:
        return self.model.lower() in TOOLSET_MODELS

    def _as_computer_call(self, block: dict) -> dict:
        """Toolset call (name=<action>, input={...}) -> computer-tool form (input={"action": <action>, ...}), as a copy."""
        if self._uses_toolset() and "action" not in (block.get("input") or {}):
            return {**block, "input": {"action": block.get("name"), **(block.get("input") or {})}}
        return block

    # ── Private helpers ─────────────────────────────────────────────────

    def _resize_screenshot(self, screenshot_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(screenshot_bytes))
        resized = img.resize((1280, 720), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        return buf.getvalue()

    def _create_client(self, Anthropic, AnthropicBedrock, AnthropicVertex):
        if self.provider == "anthropic":
            # SDK-level retries are disabled; predict() owns the shared retry budget.
            client_kwargs: Dict[str, Any] = {"api_key": self.api_key, "max_retries": 0}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            return Anthropic(**client_kwargs).with_options(
                default_headers={"anthropic-beta": COMPUTER_USE_BETA_FLAG}
            )
        elif self.provider == "bedrock":
            return AnthropicBedrock(
                aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_region=os.getenv("AWS_DEFAULT_REGION"),
                max_retries=0,
            )
        elif self.provider == "vertex":
            return AnthropicVertex(max_retries=0)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _thinking_config(self) -> Tuple[dict, int]:
        if self.no_thinking:
            return {}, self.max_tokens

        # The shared reasoning tier is passed as adaptive-thinking effort.
        if self.reasoning_effort:
            return {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": self.reasoning_effort},
            }, self.max_tokens

        model_lower = self.model.lower()
        if "opus-4-7" in model_lower:
            effort = os.environ.get("CLAUDE_ADAPTIVE_EFFORT", "low").strip().lower()
            if effort not in {"low", "medium", "high"}:
                effort = "low"
            return {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
            }, self.max_tokens

        budget_tokens = 2048
        actual_max = max(self.max_tokens, budget_tokens + 500)
        extra = {"thinking": {"type": "enabled", "budget_tokens": budget_tokens}}
        return extra, actual_max

    def _add_tool_results(self, screenshot_bytes: bytes) -> None:
        """Add tool_result messages for all tool_use blocks in the last assistant message."""
        if not self.messages:
            return
        last = self.messages[-1]
        if last.get("role") != "assistant":
            return

        tool_blocks = [self._as_computer_call(b) for b in last["content"] if b.get("type") == "tool_use"]
        for i, block in enumerate(tool_blocks):
            content = [{"type": "text", "text": "Success"}]
            # zoom: return the requested region of the current (1280x720) view, enlarged; no desktop action
            if (block.get("input") or {}).get("action") == "zoom":
                zoomed = self._zoom_region(screenshot_bytes, (block.get("input") or {}).get("region"))
                if zoomed is not None:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(zoomed).decode("utf-8"),
                            },
                        }
                    )
                else:
                    content = [{"type": "text", "text": "Error: invalid zoom region"}]
            # Attach screenshot to the last tool result (a zoom result carries only the zoomed image, as in OSWorld)
            if i == len(tool_blocks) - 1 and (block.get("input") or {}).get("action") != "zoom":
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(screenshot_bytes).decode("utf-8"),
                        },
                    }
                )
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": content,
                            # a toolset member call must be answered with its toolset_name
                            **({"toolset_name": block["toolset_name"]} if block.get("toolset_name") else {}),
                        }
                    ],
                }
            )

    @staticmethod
    def _zoom_region(screenshot_bytes: bytes, region) -> Optional[bytes]:
        """Crop region [x1, y1, x2, y2] (1280x720 view coordinates) and enlarge it to fit 1280x720."""
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in region]
        except (TypeError, ValueError):
            return None
        img = Image.open(io.BytesIO(screenshot_bytes))
        x1, x2 = sorted((max(0, min(x1, img.width)), max(0, min(x2, img.width))))
        y1, y2 = sorted((max(0, min(y1, img.height)), max(0, min(y2, img.height))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        crop = img.crop((x1, y1, x2, y2))
        scale = min(1280 / crop.width, 720 / crop.height)
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()

    def _filter_old_images(self, keep: int) -> None:
        """Keep only the `keep` most recent screenshots, oldest dropped first.

        Screenshots live either directly in the first user message or inside
        tool_result blocks; both count toward the window.
        """
        containers = []  # content lists holding screenshots, in chronological order
        for msg in self.messages:
            if not isinstance(msg["content"], list):
                continue
            containers.append(msg["content"])
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    if isinstance(item.get("content"), list):
                        containers.append(item["content"])
        total = sum(
            1
            for content in containers
            for c in content
            if isinstance(c, dict) and c.get("type") == "image"
        )
        to_remove = total - keep
        if to_remove <= 0:
            return

        for content in containers:
            for c in list(content):
                if to_remove <= 0:
                    return
                if isinstance(c, dict) and c.get("type") == "image":
                    content.remove(c)
                    to_remove -= 1

    def _response_to_params(self, response) -> list:
        """Convert API response to storable message params."""
        from anthropic.types.beta import BetaTextBlock, BetaTextBlockParam

        res = []
        for block in response.content or []:
            if isinstance(block, BetaTextBlock):
                if block.text:
                    res.append(BetaTextBlockParam(type="text", text=block.text))
                elif getattr(block, "type", None) == "thinking":
                    thinking_block = {
                        "type": "thinking",
                        "thinking": getattr(block, "thinking", None),
                    }
                    if hasattr(block, "signature"):
                        thinking_block["signature"] = getattr(block, "signature", None)
                    res.append(thinking_block)
            else:
                res.append(cast(dict, block.model_dump()))
        return res

    def _extract_raw_response_string(self, response) -> str:
        """Extract and concatenate raw response content into a single string."""
        raw_response_str = ""
        if response.content:
            for block in response.content:
                if hasattr(block, 'text') and block.text:
                    raw_response_str += f"[TEXT] {block.text}\n"
                elif hasattr(block, 'thinking') and block.thinking:
                    raw_response_str += f"[THINKING] {block.thinking}\n"
                elif hasattr(block, 'name') and hasattr(block, 'input'):
                    raw_response_str += f"[TOOL_USE] {block.name}: {block.input}\n"
                else:
                    raw_response_str += f"[OTHER] {str(block)}\n"
        return raw_response_str.strip()

    def _parse_response(
        self, response_params: list, raw_response
    ) -> Tuple[str, List[str]]:
        """Extract reasoning text and action commands from response params."""
        raw_response_str = self._extract_raw_response_string(raw_response)

        actions: List[str] = []
        reasonings: List[str] = []

        for block in response_params:
            if block.get("type") == "tool_use":
                cmd = self._parse_actions_from_tool_call(self._as_computer_call(block))
                actions.append(cmd)
            elif block.get("type") == "text":
                reasonings.append(block["text"])

        reasoning = reasonings[0] if reasonings else ""

        if raw_response_str and "[INFEASIBLE]" in raw_response_str:
            return reasoning, ["FAIL"]

        if not actions:
            return reasoning, ["DONE"]

        return reasoning, actions

    def _parse_actions_from_tool_call(self, tool_call: dict) -> str:
        """Convert a Claude computer-use tool_call to pyautogui code string.

        Ported from OSWorld/mm_agents/anthropic/main.py parse_actions_from_tool_call.
        """
        import ast  # For parsing string coordinates from bltcy API

        result = ""
        function_args = tool_call.get("input", {})

        # ========== bltcy API compatibility fix ==========
        # Fix 1: Parse string coordinates to arrays
        # bltcy returns coordinate: "[100, 100]" (string), we need [100, 100] (array)
        for coord_key in ["coordinate", "start_coordinate"]:
            coord_value = function_args.get(coord_key)
            if coord_value and isinstance(coord_value, str):
                try:
                    parsed = ast.literal_eval(coord_value)
                    if isinstance(parsed, (list, tuple)):
                        function_args[coord_key] = list(parsed)
                        logger.debug(f"Parsed {coord_key} from string: {coord_value} -> {function_args[coord_key]}")
                    else:
                        logger.warning(f"{coord_key} parsed to non-list type: {type(parsed)}")
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Failed to parse {coord_key}: {coord_value} - {e}")

        # Fix 2: Map action names for bltcy compatibility
        # bltcy returns "click", code expects "left_click"
        action = function_args.get("action")
        if not action:
            action = function_args.get("name", "")

        bltcy_action_conversion = {
            "click": "left_click",  # bltcy -> Anthropic format
        }
        action = bltcy_action_conversion.get(action, action)
        function_args["action"] = action  # Update for downstream code

        # Fix 3: Map parameter names for bltcy compatibility
        # bltcy returns "key" parameter for key actions, code expects "text"
        # Extended to handle all text-based actions (key, type, hold_key)
        if action in ('key', 'type', 'hold_key'):
            if 'text' not in function_args:
                # Try alternative field names that bltcy might use
                possible_fields = ['key', 'keys', 'content', 'string', 'value']
                for alt_field in possible_fields:
                    if alt_field in function_args:
                        value = function_args.pop(alt_field)
                        if isinstance(value, str) and value.strip():
                            function_args['text'] = value
                            logger.debug(f"Mapped bltcy field '{alt_field}' -> 'text': '{value}' for action '{action}'")
                            break
                        else:
                            logger.warning(f"Field '{alt_field}' exists but is empty for {action}")

                # If still no valid text, FAIL explicitly
                if 'text' not in function_args or not function_args['text'].strip():
                    logger.error(f"Cannot execute {action}: no valid text field found. bltcy response: {function_args}")
                    return "FAIL"
        # ========== End bltcy fix ==========

        action_conversion = {
            "left click": "click",
            "right click": "right_click",
        }
        action = action_conversion.get(action, action)

        text = function_args.get("text")
        coordinate = function_args.get("coordinate")
        start_coordinate = function_args.get("start_coordinate")
        scroll_direction = function_args.get("scroll_direction")
        scroll_amount = function_args.get("scroll_amount")
        duration = function_args.get("duration")

        def scale(coord):
            if coord is None:
                return None
            return (
                int(coord[0] * self.resize_factor[0]),
                int(coord[1] * self.resize_factor[1]),
            )

        coordinate = scale(coordinate)
        start_coordinate = scale(start_coordinate)

        if action == "left_mouse_down":
            result += "pyautogui.mouseDown()\n"
        elif action == "left_mouse_up":
            result += "pyautogui.mouseUp()\n"

        elif action == "hold_key":
            if not isinstance(text, str):
                raise ValueError(f"{text} must be a string")
            keys = text.split('+')
            for key in keys:
                key = key.strip().lower()
                result += f"pyautogui.keyDown('{key}')\n"

        elif action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ValueError(f"coordinate is required for {action}")
            if text is not None:
                raise ValueError(f"text is not accepted for {action}")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise ValueError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) for i in coordinate):
                raise ValueError(f"{coordinate} must be a tuple of ints")

            x, y = coordinate[0], coordinate[1]
            if action == "mouse_move":
                result += f"pyautogui.moveTo({x}, {y}, duration={duration or 0.5})\n"
            elif action == "left_click_drag":
                if start_coordinate:
                    if not isinstance(start_coordinate, (list, tuple)) or len(start_coordinate) != 2:
                        raise ValueError(f"{start_coordinate} must be a tuple of length 2")
                    if not all(isinstance(i, int) for i in start_coordinate):
                        raise ValueError(f"{start_coordinate} must be a tuple of ints")
                    start_x, start_y = start_coordinate[0], start_coordinate[1]
                    result += f"pyautogui.moveTo({start_x}, {start_y}, duration={duration or 0.5})\n"
                result += f"pyautogui.dragTo({x}, {y}, duration={duration or 0.5})\n"

        elif action in ("key", "type"):
            if text is None:
                raise ValueError(f"text is required for {action}")
            if coordinate is not None:
                raise ValueError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ValueError(f"{text} must be a string")

            if action == "key":
                key_conversion = {
                    "page_down": "pagedown",
                    "page_up": "pageup",
                    "super_l": "win",
                    "super": "command",
                    "escape": "esc",
                }
                keys = text.split('+')
                for key in keys:
                    key = key.strip().lower()
                    key = key_conversion.get(key, key)
                    result += f"pyautogui.keyDown('{key}')\n"
                for key in reversed(keys):
                    key = key.strip().lower()
                    key = key_conversion.get(key, key)
                    result += f"pyautogui.keyUp('{key}')\n"
            elif action == "type":
                for char in text:
                    if char == '\n':
                        result += "pyautogui.press('enter')\n"
                    elif char == "'":
                        result += 'pyautogui.press("\'")\n'
                    elif char == '\\':
                        result += "pyautogui.press('\\\\')\n"
                    elif char == '"':
                        result += "pyautogui.press('\"')\n"
                    else:
                        result += f"pyautogui.press('{char}')\n"

        elif action == "scroll":
            if text is not None:
                result += f"pyautogui.keyDown('{text.lower()}')\n"
            if coordinate is None:
                if scroll_direction in ("up", "down"):
                    result += f"pyautogui.scroll({scroll_amount if scroll_direction == 'up' else -scroll_amount})\n"
                elif scroll_direction in ("left", "right"):
                    result += f"pyautogui.hscroll({scroll_amount if scroll_direction == 'right' else -scroll_amount})\n"
            else:
                x, y = coordinate[0], coordinate[1]
                if scroll_direction in ("up", "down"):
                    result += f"pyautogui.scroll({scroll_amount if scroll_direction == 'up' else -scroll_amount}, {x}, {y})\n"
                elif scroll_direction in ("left", "right"):
                    result += f"pyautogui.hscroll({scroll_amount if scroll_direction == 'right' else -scroll_amount}, {x}, {y})\n"
            if text is not None:
                result += f"pyautogui.keyUp('{text.lower()}')\n"

        elif action in ("left_click", "right_click", "double_click", "middle_click", "left_press", "triple_click"):
            if text:
                keys = text.split('+')
                for key in keys:
                    key = key.strip().lower()
                    result += f"pyautogui.keyDown('{key}')\n"
            if coordinate is not None:
                x, y = coordinate
                if action == "left_click":
                    result += f"pyautogui.click({x}, {y})\n"
                elif action == "right_click":
                    result += f"pyautogui.rightClick({x}, {y})\n"
                elif action == "double_click":
                    result += f"pyautogui.doubleClick({x}, {y})\n"
                elif action == "middle_click":
                    result += f"pyautogui.middleClick({x}, {y})\n"
                elif action == "left_press":
                    result += f"pyautogui.mouseDown({x}, {y})\n"
                    result += "time.sleep(1)\n"
                    result += f"pyautogui.mouseUp({x}, {y})\n"
                elif action == "triple_click":
                    result += f"pyautogui.tripleClick({x}, {y})\n"
            else:
                if action == "left_click":
                    result += "pyautogui.click()\n"
                elif action == "right_click":
                    result += "pyautogui.rightClick()\n"
                elif action == "double_click":
                    result += "pyautogui.doubleClick()\n"
                elif action == "middle_click":
                    result += "pyautogui.middleClick()\n"
                elif action == "left_press":
                    result += "pyautogui.mouseDown()\n"
                    result += "time.sleep(1)\n"
                    result += "pyautogui.mouseUp()\n"
                elif action == "triple_click":
                    result += "pyautogui.tripleClick()\n"
            if text:
                keys = text.split('+')
                for key in reversed(keys):
                    key = key.strip().lower()
                    result += f"pyautogui.keyUp('{key}')\n"

        elif action == "wait":
            result += "WAIT"
        elif action == "fail":
            result += "FAIL"
        elif action == "done":
            result += "DONE"
        elif action == "call_user":
            result += "FAIL"
        elif action == "screenshot":
            result += "WAIT"
        elif action == "zoom":
            # view-only: the zoomed image is returned in the next tool_result (_add_tool_results)
            result += "WAIT"
        else:
            raise ValueError(f"Invalid action: {action}")

        return result
