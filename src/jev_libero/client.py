"""Jev decisions through OpenRouter, TypeSafe, or the BXI Responses gateway."""

import json
import os
import time
from pathlib import Path

import requests

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
PROVIDERS = {
    "openrouter": (ENDPOINT, MODEL, "OPENROUTER"),
    "typesafe": ("https://api.typesafe.ai/v1/systemone", "jev-latest", "TYPESAFE"),
    "bxi": (os.environ["BXI_BASE_URL"], os.environ.get("BXI_MODEL", "gpt-5.6-sol"), "BXI"),
}
# Official published price: https://typesafe.ai ($42/billion input tokens).
TYPESAFE_INPUT_USD_PER_MILLION = 0.042


class BudgetExceeded(RuntimeError):
    pass


def append_json(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def api_key(key_file=None, provider="openrouter"):
    prefix = PROVIDERS[provider][2]
    file_var, key_var = f"{prefix}_API_KEY_FILE", f"{prefix}_API_KEY"
    if key_file or os.environ.get(file_var):
        value = Path(key_file or os.environ[file_var]).expanduser().read_text().strip()
    else:
        value = os.environ.get(key_var, "").strip()
    if not value:
        raise ValueError(f"Set {key_var} or {file_var} (never commit credentials).")
    return value


class Decisions:
    def __init__(
        self, out, budget_usd=0.10, key_file=None, model=None, session=None, provider="openrouter"
    ):
        self.out = Path(out)
        self.budget_usd = budget_usd
        self.provider = provider
        self.endpoint, default_model, _ = PROVIDERS[provider]
        self.model = model or default_model
        self.total = 0.0
        self.calls = 0
        self.session = session if session is not None else requests.Session()
        self.session.headers["Authorization"] = "Bearer " + api_key(key_file, provider)

    def choose(self, step, layer, state, instructions, criteria):
        if self.total + 0.005 > self.budget_usd:
            raise BudgetExceeded(
                f"API cost guard reached (${self.total:.6f}, budget ${self.budget_usd:.2f})"
            )
        body = {
            "model": self.model,
            "state": state,
            "questions": {
                layer: {"type": "choice", "instructions": instructions, "criteria": criteria}
            },
        }
        if self.provider == "bxi":
            body = {"model": self.model, "stream": True, "input": [{"role": "user", "content": [{
                "type": "input_text",
                "text": ("Return JSON only as {\\\"choice\\\":\\\"...\\\"}. "
                         f"The choice must be one of: {list(criteria)}. Instructions: {instructions}. "
                         f"State: {json.dumps(state, ensure_ascii=False)}"),
            }]}]}
        for attempt in range(2):
            start = time.perf_counter()
            try:
                response = self.session.post(self.endpoint, json=body, timeout=45)
                break
            except requests.exceptions.SSLError as exc:
                append_json(
                    self.out / "transport_errors.jsonl",
                    {
                        "step": step,
                        "layer": layer,
                        "attempt": attempt,
                        "request": body,
                        "error": str(exc),
                    },
                )
                if attempt:
                    raise
                time.sleep(2)
        elapsed = time.perf_counter() - start
        try:
            result = self._parse_response(response, layer) if self.provider == "bxi" else response.json()
        except ValueError:
            append_json(
                self.out / "api.jsonl",
                {
                    "step": step,
                    "layer": layer,
                    "request": body,
                    "http_status": response.status_code,
                    "response_text": response.text,
                    "latency_s": elapsed,
                },
            )
            response.raise_for_status()
            raise
        # Only body and response are logged; never request headers or credentials.
        append_json(
            self.out / "api.jsonl",
            {
                "step": step,
                "layer": layer,
                "request": body,
                "response": result,
                "http_status": response.status_code,
                "latency_s": elapsed,
            },
        )
        response.raise_for_status()
        if self.provider == "typesafe":
            cost = result["usage"]["input_tokens"] * TYPESAFE_INPUT_USD_PER_MILLION / 1_000_000
            append_json(
                self.out / "cost_estimates.jsonl",
                {
                    "step": step,
                    "layer": layer,
                    "estimated_cost_usd": cost,
                    "input_usd_per_million": TYPESAFE_INPUT_USD_PER_MILLION,
                },
            )
        elif self.provider == "bxi":
            cost = result["usage"]["total_tokens"] * TYPESAFE_INPUT_USD_PER_MILLION / 1_000_000
        else:
            cost = result["usage"]["cost"]
        self.total += cost
        self.calls += 1
        choice = result["answers"][layer]["choice"]
        if choice not in criteria:
            raise ValueError(f"Invalid {layer} choice returned: {choice}")
        return choice

    @staticmethod
    def _parse_response(response, layer):
        text, usage = "", {"total_tokens": 0}
        for line in response.text.splitlines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event.get("type") == "response.output_text.delta":
                text += event.get("delta", "")
            elif event.get("type") == "response.completed":
                usage = event.get("response", {}).get("usage", usage) or usage
            elif event.get("type") == "error":
                raise RuntimeError(event.get("error", {}).get("message", "BXI response failed"))
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"BXI returned non-JSON choice for {layer}: {text!r}") from exc
        return {"answers": {layer: {"choice": payload.get("choice")}}, "usage": usage}

    def close(self):
        self.session.close()
