"""LLM-style report synthesizer.

This module provides the interface for DeepSeek-style medical report writing
without forcing a hard dependency on an external API during local runs.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class DeepSeekReportSynthesizer:
    """Fallback report generator that mirrors a clinical summary format."""

    def __init__(self, api_key: str | None = None, endpoint: str | None = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.endpoint = endpoint or os.getenv("DEEPSEEK_API_URL")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.is_configured = bool(self.api_key and self.endpoint)

    def _call_remote(self, prompt: str) -> str | None:
        if not self.is_configured:
            return None

        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                body = json.loads(response.read().decode("utf-8") or "{}")
                choices = body.get("choices", [])
                if not choices:
                    return None
                message = choices[0].get("message", {})
                response_text = message.get("content") or ""
                return response_text.strip() or None
        except Exception as exc:
            print(f"[llm_report] DeepSeek call failed, falling back: {exc}")
            return None

    def generate(self, summary: str, features: dict[str, Any] | None = None) -> dict[str, Any]:
        prompt = self._build_prompt(summary, features)
        response_text = self._call_remote(prompt)
        if response_text:
            return {
                "summary": response_text,
                "status": "ready",
                "format": "medical_style_report",
                "note": "DeepSeek API response used for final medical report synthesis.",
                **({"features": features} if features else {}),
            }

        base = {
            "summary": summary,
            "status": "fallback",
            "format": "medical_style_report",
            "note": (
                "DeepSeek is not configured; the project is using the local structured report template."
            ),
        }
        if features:
            base["features"] = features
        return base

    @staticmethod
    def _build_prompt(summary: str, features: dict[str, Any] | None = None) -> str:
        detail = ""
        if features:
            detail = "\n".join(
                f"{key}: {value}"
                for key, value in features.items()
                if value is not None
            )
        return (
            "Write a concise, medically structured melanoma assessment report in plain English. "
            "Keep it informative but non-diagnostic and emphasize imaging and ABCD evidence.\n\n"
            f"Summary context:\n{summary}\n\n"
            f"Evidence:\n{detail or 'No additional structured evidence provided.'}"
        )


llm_report_synthesizer = DeepSeekReportSynthesizer()
