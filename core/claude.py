"""
core/claude.py — LLM backend wrapper
=====================================
Currently configured to use a LOCAL OLLAMA model via the OpenAI-compatible API.

TO SWITCH BACK TO CLAUDE (Anthropic API):
------------------------------------------
1. Install the Anthropic SDK:
       pip install anthropic

2. In your .env, set:
       ANTHROPIC_API_KEY=your_real_key_here
       CLAUDE_MODEL=claude-sonnet-4-5        # or any model you want

3. In main.py, change:
       claude_model    = os.getenv("OLLAMA_MODEL", "")
       ollama_base_url = os.getenv("OLLAMA_BASE_URL", "")
   back to:
       claude_model      = os.getenv("CLAUDE_MODEL", "")
       anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")

4. In this file (claude.py):
   a) Replace the import at the top:
          from openai import OpenAI           # OLLAMA (current)
      with:
          from anthropic import Anthropic     # CLAUDE
          from anthropic.types import Message # also remove our custom Message class below

   b) Replace the client inside __init__:
          self.client = OpenAI(base_url=..., api_key="ollama")   # OLLAMA (current)
      with:
          self.client = Anthropic()   # reads ANTHROPIC_API_KEY from env automatically

   c) Replace the entire OpenAI response-parsing block at the bottom of chat()
      with just:
          message = self.client.messages.create(**params)
          return message
      The Anthropic SDK returns a real Message — no adapter needed.

   d) In add_user_message(), change the tool-result branch to Anthropic format:
          messages.append({
              "role": "user",
              "content": [{"type": "tool_result", "tool_use_id": item["tool_use_id"], "content": item["content"]}
                          for item in message],
          })

   e) In add_assistant_message(), simplify to:
          messages.append({"role": "assistant", "content": message.content})

   f) In tools.py, change the tool-block check from:
          isinstance(block, ToolUseBlock)
      back to:
          block.type == "tool_use"

   g) Delete the three adapter classes (TextBlock, ToolUseBlock, Message) below.

5. chat.py and cli_chat.py need NO changes.
"""

import os
import json

# ── OLLAMA: uses OpenAI-compatible client pointed at local Ollama server ──────
# TO USE CLAUDE: replace with `from anthropic import Anthropic`
from openai import OpenAI


# ── Adapter classes ────────────────────────────────────────────────────────────
# These exist ONLY because we swapped from Anthropic SDK to OpenAI SDK.
# The Anthropic SDK returns real Message / TextBlock / ToolUseBlock objects.
# When switching back to Claude, delete these three classes and import
# Message directly from `anthropic.types`.

class TextBlock:
    """Mimics anthropic.types.TextContentBlock"""
    type = "text"
    def __init__(self, text: str):
        self.text = text


class ToolUseBlock:
    """Mimics anthropic.types.ToolUseBlock"""
    type = "tool_use"
    def __init__(self, id: str, name: str, input: dict):
        self.id = id
        self.name = name
        self.input = input


class Message:
    """
    Mimics anthropic.types.Message.
    When switching back to Claude, remove this and import:
        from anthropic.types import Message
    """
    def __init__(self, content: list, stop_reason: str):
        self.content = content
        self.stop_reason = stop_reason


# ── LLM service class ──────────────────────────────────────────────────────────

class Claude:
    def __init__(self, model: str):
        # ── OLLAMA setup ──────────────────────────────────────────────────────
        # Ollama exposes an OpenAI-compatible API at localhost:11434/v1.
        # The api_key value is required by the openai lib but ignored by Ollama.
        #
        # TO USE CLAUDE: replace this block with:
        #     self.client = Anthropic()
        #     self.model = model
        self.client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )
        self.model = model

    def add_user_message(self, messages: list, message):
        """
        Appends a user turn to the messages list.

        NOTE — tool result format differs between providers:
          OLLAMA / OpenAI:   role="tool" with tool_call_id
          CLAUDE / Anthropic: role="user" with content type="tool_result"

        TO USE CLAUDE: replace the `elif isinstance(message, list)` branch with:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": item["tool_use_id"],
                        "content": item["content"],
                    }
                    for item in message
                ],
            })
        """
        if isinstance(message, Message):
            # Convert content blocks back to plain text for OpenAI format
            text = "\n".join(
                b.text for b in message.content if isinstance(b, TextBlock)
            )
            messages.append({"role": "user", "content": text})
        elif isinstance(message, list):
            # Tool results — OpenAI/Ollama format: one message per result with role="tool"
            for item in message:
                messages.append({
                    "role": "tool",
                    "tool_call_id": item["tool_use_id"],
                    "content": item["content"],
                })
        else:
            messages.append({"role": "user", "content": message})

    def add_assistant_message(self, messages: list, message):
        """
        Appends an assistant turn to the messages list.

        TO USE CLAUDE: simplify this entire method to:
            messages.append({
                "role": "assistant",
                "content": message.content,  # Anthropic content blocks, already correct type
            })
        """
        if isinstance(message, Message):
            tool_calls = [b for b in message.content if isinstance(b, ToolUseBlock)]
            text_blocks = [b for b in message.content if isinstance(b, TextBlock)]

            if tool_calls:
                # OpenAI/Ollama format: tool_calls is a separate key on the message
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(b.text for b in text_blocks) or None,
                    "tool_calls": [
                        {
                            "id": b.id,
                            "type": "function",
                            "function": {
                                "name": b.name,
                                "arguments": json.dumps(b.input),
                            },
                        }
                        for b in tool_calls
                    ],
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(b.text for b in text_blocks),
                })
        else:
            messages.append({"role": "assistant", "content": str(message)})

    def text_from_message(self, message: Message) -> str:
        """Extracts plain text from a Message. No changes needed when switching to Claude."""
        return "\n".join(
            b.text for b in message.content if isinstance(b, TextBlock)
        )

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,        # NOTE: not supported by Ollama; works with Claude 3.7+ models
        thinking_budget=1024,  # NOTE: not supported by Ollama; works with Claude 3.7+ models
    ) -> Message:
        """
        Sends a chat request and returns a Message.

        TO USE CLAUDE: replace the OpenAI call + response-parsing block at the
        bottom of this method with:
            message = self.client.messages.create(**params)
            return message   # Anthropic already returns a proper Message object

        Also restore the thinking block inside params if needed:
            if thinking:
                params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        And change the stop key back:
            params["stop_sequences"] = stop_sequences   (not "stop")
        """
        formatted_messages = []
        if system:
            # OpenAI/Ollama: system prompt goes as first message with role="system"
            # CLAUDE: system prompt is a top-level `system` key — move it to params["system"]
            formatted_messages.append({"role": "system", "content": system})
        formatted_messages.extend(messages)

        params = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
        }

        if tools:
            # ── OLLAMA: tools use OpenAI function-calling schema ──────────────
            # TO USE CLAUDE: replace this block with just:
            #     params["tools"] = tools   (Anthropic accepts input_schema directly)
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                }
                for t in tools
            ]
            params["tool_choice"] = "auto"

        if stop_sequences:
            # OpenAI/Ollama uses "stop"; Anthropic uses "stop_sequences"
            # TO USE CLAUDE: change "stop" back to "stop_sequences"
            params["stop"] = stop_sequences

        # ── OLLAMA: call via OpenAI-compatible API ────────────────────────────
        # TO USE CLAUDE: replace everything below (up to `return`) with:
        #     message = self.client.messages.create(**params)
        #     return message
        response = self.client.chat.completions.create(**params)
        choice = response.choices[0]

        # Parse OpenAI response into our adapter Message object
        content_blocks = []

        if choice.message.content:
            content_blocks.append(TextBlock(choice.message.content))

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                content_blocks.append(ToolUseBlock(
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                ))

        # Map OpenAI finish_reason → Anthropic-style stop_reason
        # CLAUDE returns stop_reason directly on the Message object; no mapping needed
        stop_reason = (
            "tool_use" if choice.finish_reason == "tool_calls" else "end_turn"
        )

        return Message(content=content_blocks, stop_reason=stop_reason)