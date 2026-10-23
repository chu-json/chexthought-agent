"""Model backends behind one interface: `call(system, parts, role) -> Response`.

`parts` is a list of PIL images and text strings, sent in order as a single user message.
"""
import base64
import hashlib
import io
import json
import random
from dataclasses import dataclass

from . import boxes as bx
from .data import FINDINGS


@dataclass
class Response:
    text: str
    stop_reason: str  # "refusal" marks a refusal on every backend
    input_tokens: int
    output_tokens: int
    raw: dict


def image_digest(im):
    return hashlib.sha256(f"{im.mode}{im.size}".encode() + im.tobytes()).hexdigest()


def request_hash(system, parts, settings):
    """Hash of everything that determines a response: prompt text, image pixels and model settings."""
    desc = {
        "system": system,
        "parts": [p if isinstance(p, str) else {"image": image_digest(p)} for p in parts],
        "settings": settings,
    }
    return hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest()


def png_base64(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


class Backend:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg["model"]

    def settings(self):
        """Sampling and thinking settings, logged with every call and part of the cache key."""
        keys = ("backend", "model", "max_tokens", "temperature", "thinking", "effort")
        return {k: self.cfg.get(k) for k in keys}

    def call(self, system, parts, role):
        raise NotImplementedError


class AnthropicBackend(Backend):
    """Claude Messages API. temperature must be null: current Claude models reject sampling parameters."""

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.get("temperature") is not None:
            raise ValueError("Claude models reject temperature; set it to null (API default)")
        import anthropic
        self.client = anthropic.Anthropic(max_retries=cfg.get("max_retries", 5), timeout=cfg.get("timeout", 300))

    def request_kwargs(self, system, parts):
        """The exact arguments sent to messages.create."""
        content = [
            {"type": "text", "text": p} if isinstance(p, str)
            else {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_base64(p)}}
            for p in parts
        ]
        kwargs = {
            "model": self.model, "max_tokens": self.cfg["max_tokens"], "system": system,
            "messages": [{"role": "user", "content": content}],
        }
        if self.cfg.get("thinking"):
            kwargs["thinking"] = {"type": self.cfg["thinking"]}  # "adaptive" or "between_tools" (no thinking)
        if self.cfg.get("effort"):
            kwargs["output_config"] = {"effort": self.cfg["effort"]}
        return kwargs

    def call(self, system, parts, role):
        r = self.client.messages.create(**self.request_kwargs(system, parts))
        text = "".join(b.text for b in r.content if b.type == "text")
        return Response(text, r.stop_reason, r.usage.input_tokens, r.usage.output_tokens, r.to_dict())


def openai_messages(system, parts):
    """Chat Completions messages with images as base64 data URLs."""
    content = [
        {"type": "text", "text": p} if isinstance(p, str)
        else {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_base64(p)}"}}
        for p in parts
    ]
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


class OpenAIBackend(Backend):
    """OpenAI Chat Completions. Implemented, unused in v0."""

    max_tokens_param = "max_completion_tokens"

    def __init__(self, cfg):
        super().__init__(cfg)
        from openai import OpenAI
        self.client = OpenAI(max_retries=cfg.get("max_retries", 5), **self.client_kwargs())

    def client_kwargs(self):
        return {}

    def call(self, system, parts, role):
        kwargs = {"model": self.model, "messages": openai_messages(system, parts),
                  self.max_tokens_param: self.cfg["max_tokens"]}
        if self.cfg.get("temperature") is not None:
            kwargs["temperature"] = self.cfg["temperature"]
        if self.cfg.get("effort"):
            kwargs["reasoning_effort"] = self.cfg["effort"]
        r = self.client.chat.completions.create(**kwargs)
        msg = r.choices[0].message
        stop = "refusal" if getattr(msg, "refusal", None) else r.choices[0].finish_reason
        return Response(msg.content or "", stop, r.usage.prompt_tokens, r.usage.completion_tokens, r.model_dump())


class OpenAICompatibleBackend(OpenAIBackend):
    """OpenAI-compatible server, e.g. Qwen3-VL or MedGemma served with vLLM. Stub: untested."""

    max_tokens_param = "max_tokens"

    def client_kwargs(self):
        return {"base_url": self.cfg["base_url"], "api_key": self.cfg.get("api_key", "EMPTY")}


class MockBackend(Backend):
    """Deterministic fake outputs for tests, seeded by a hash of the request.

    Occasionally returns broken JSON or a refusal so those code paths get exercised.
    """

    invalid_rate = 0.1
    refusal_rate = 0.02

    def call(self, system, parts, role):
        rng = random.Random(request_hash(system, parts, self.settings()))
        images = [p for p in parts if not isinstance(p, str)]
        prompt = "\n".join(p for p in parts if isinstance(p, str))
        if rng.random() < self.refusal_rate:
            text, stop = "", "refusal"
        elif rng.random() < self.invalid_rate:
            text, stop = "Here is my report: {not valid json", "end_turn"
        elif role == "helper":
            text, stop = self._helper(rng, *images[0].size), "end_turn"
        else:
            text, stop = self._reader(rng, prompt), "end_turn"
        in_tokens = (len(system) + len(prompt)) // 4 + 1100 * len(images)
        return Response(text, stop, in_tokens, len(text) // 4, {"mock": True})

    def _reader(self, rng, prompt):
        earlier = None
        if "Your earlier report:\n" in prompt:
            earlier = json.loads(prompt.split("Your earlier report:\n", 1)[1].split("\n\n", 1)[0])
        findings = {}
        for f in FINDINGS:
            if earlier:  # pass 2: flip some answers
                present = earlier["findings"][f]["present"]
                if rng.random() < 0.15:
                    present = "no" if present == "yes" else "yes"
            else:
                present = "yes" if rng.random() < 0.35 else "no"
            prob = rng.randint(55, 95) if present == "yes" else rng.randint(2, 45)
            findings[f] = {"present": present, "probability": prob, "location": "mock" if present == "yes" else ""}
        return json.dumps({"findings": findings, "reasoning": "mock"})

    def _helper(self, rng, width, height):
        fmt = self.cfg.get("box_format", "norm")
        out = []
        for _ in range(rng.randint(0, bx.N_BOXES)):
            w, h = rng.uniform(0.1, 0.4), rng.uniform(0.1, 0.4)
            x1, y1 = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
            native = bx.from_norm((x1, y1, x1 + w, y1 + h), fmt, width, height)
            out.append({"box": [round(v, 3) for v in native], "suspected_finding": rng.choice(FINDINGS + ["other"]),
                        "reason": "mock"})
        return json.dumps({"boxes": out})


BACKENDS = {
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "openai_compatible": OpenAICompatibleBackend,
    "mock": MockBackend,
}


def make_backend(cfg):
    return BACKENDS[cfg["backend"]](cfg)
