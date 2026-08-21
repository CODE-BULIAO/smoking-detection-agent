"""OpenAI-compatible vision verifier used as an LLM fallback for smoking alerts."""
import base64
import json
import os
import time
import urllib.request

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

PROMPT = ("你是吸烟行为复核员。请观察图中人物，判断其是否正在吸烟"
          "（手中或嘴里持有点燃的香烟或电子烟，或有明显吸烟动作）。"
          "只输出 JSON：{\"smoking\": true 或 false, \"reason\": \"一句中文依据\"}。")

BATCH_PROMPT = ("你是吸烟行为复核员。以下按顺序给出编号为 1..N 的监控截图，"
                "请逐张判断画面中的人物是否正在吸烟"
                "（手中或嘴里持有点燃的香烟或电子烟，或有明显吸烟动作）。"
                "只输出 JSON 数组，不要输出其他内容："
                "[{\"id\": 1, \"smoking\": true 或 false, \"reason\": \"一句中文依据\"}, ...]")


def load_config(base_url=None, model=None, api_key=None):
    cfg = {"base_url": base_url, "model": model, "api_key": api_key}
    path = os.path.join(DEMO_DIR, "llm_config.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                file_cfg = json.load(f)
        except Exception:
            file_cfg = {}
        for k in ("base_url", "model", "api_key"):
            cfg[k] = cfg[k] or file_cfg.get(k)
    for k, envk in (("base_url", "LLM_BASE_URL"), ("model", "LLM_MODEL"),
                    ("api_key", "LLM_API_KEY")):
        cfg[k] = cfg[k] or os.getenv(envk)
    if not (cfg["base_url"] and cfg["model"] and cfg["api_key"]):
        return None
    return cfg


class LLMVerifier:
    def __init__(self, cfg, timeout=30):
        self.cfg = cfg
        self.timeout = timeout

    def verify(self, jpeg_bytes):
        t0 = time.time()
        b64 = base64.b64encode(jpeg_bytes).decode()
        body = {
            "model": self.cfg["model"],
            "temperature": 0,
            "max_tokens": 200,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
        }
        if "qwen" in self.cfg["model"].lower():
            body["enable_thinking"] = False
        req = urllib.request.Request(
            self.cfg["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.cfg['api_key']}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)
            raw = data["choices"][0]["message"]["content"]
        except Exception as exc:
            return {"smoking": None, "reason": f"调用失败: {exc}",
                    "raw": "", "latency": round(time.time() - t0, 2), "usage": {},
                    "prompt": PROMPT}
        verdict, reason = self._parse(raw)
        return {"smoking": verdict, "reason": reason, "raw": raw,
                "latency": round(time.time() - t0, 2),
                "usage": data.get("usage") or {}, "prompt": PROMPT}

    def verify_batch(self, jpeg_list):
        """一次调用复核多张图，节省 token。失败时返回 None（调用方回退为逐张）。"""
        if len(jpeg_list) == 1:
            return [self.verify(jpeg_list[0])]
        t0 = time.time()
        content = [{"type": "text", "text": BATCH_PROMPT}]
        for i, jpeg in enumerate(jpeg_list, 1):
            b64 = base64.b64encode(jpeg).decode()
            content.append({"type": "text", "text": f"图 {i}："})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        body = {
            "model": self.cfg["model"],
            "temperature": 0,
            "max_tokens": 160 * len(jpeg_list),
            "messages": [{"role": "user", "content": content}],
        }
        if "qwen" in self.cfg["model"].lower():
            body["enable_thinking"] = False
        req = urllib.request.Request(
            self.cfg["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.cfg['api_key']}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout + 15 * len(jpeg_list)) as r:
                data = json.load(r)
            raw = data["choices"][0]["message"]["content"]
        except Exception as exc:
            return [{"smoking": None, "reason": f"调用失败: {exc}", "raw": "",
                     "latency": round(time.time() - t0, 2), "usage": {},
                     "prompt": BATCH_PROMPT} for _ in jpeg_list]
        latency = round(time.time() - t0, 2)
        usage = data.get("usage") or {}
        try:
            start, end = raw.index("["), raw.rindex("]")
            arr = json.loads(raw[start:end + 1])
            by_id = {it.get("id"): it for it in arr if isinstance(it, dict)}
            results = []
            for i in range(len(jpeg_list)):
                item = by_id.get(i + 1) or (arr[i] if i < len(arr) and isinstance(arr[i], dict) else {})
                results.append({"smoking": bool(item.get("smoking")) if item else None,
                                "reason": str(item.get("reason", "")) if item else "解析失败",
                                "raw": raw, "latency": latency, "usage": usage,
                                "prompt": BATCH_PROMPT})
            return results
        except Exception:
            return None

    @staticmethod
    def _parse(raw):
        try:
            start, end = raw.index("{"), raw.rindex("}")
            obj = json.loads(raw[start:end + 1])
            return bool(obj.get("smoking")), str(obj.get("reason", ""))
        except Exception:
            return ("true" in raw.lower()), raw[:200]
