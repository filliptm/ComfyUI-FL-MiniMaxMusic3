import re
from datetime import datetime, timezone
from pathlib import Path


STEP_PATTERN = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")
LOSS_PATTERN = re.compile(r"(?:step[_ ]?loss|loss)\s*[=:]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.IGNORECASE)
LR_PATTERN = re.compile(r"(?:learning[_ ]?rate|lr)\s*[=:]\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.IGNORECASE)


class LogProgress:
    def __init__(self, path, total):
        self.path = Path(path)
        self.total = int(total)
        self.offset = 0
        self.partial = ""
        self.phase = "preflight"
        self.step = 0
        self.loss = None
        self.learning_rate = None

    def _phase(self, line):
        lower = line.lower()
        if "validation" in lower or "generating audio" in lower:
            return "validating"
        if "text embed" in lower or "text cache" in lower:
            return "caching_text"
        if "vae cache" in lower or "audio cache" in lower or "encoding audio" in lower:
            return "caching_audio"
        if "saving" in lower and ("checkpoint" in lower or "lora" in lower):
            return "exporting"
        if "training" in lower or ("epoch" in lower and "step" in lower) or LOSS_PATTERN.search(line):
            return "training"
        if "download" in lower or "huggingface" in lower:
            return "downloading_models"
        return self.phase

    def poll(self):
        if not self.path.is_file():
            return None
        with self.path.open("r", encoding="utf-8", errors="replace") as file:
            file.seek(self.offset)
            chunk = file.read()
            self.offset = file.tell()
        if not chunk:
            return None
        text = self.partial + chunk
        parts = re.split(r"[\r\n]", text)
        self.partial = parts.pop() if parts else ""
        changed = False
        message = ""
        for line in parts:
            line = line.strip()
            if not line:
                continue
            message = line[-500:]
            phase = self._phase(line)
            if phase != self.phase:
                self.phase = phase
                changed = True
            steps = STEP_PATTERN.findall(line)
            if steps and phase == "training":
                current, total = map(int, steps[-1])
                if current <= total and total > 0:
                    if current != self.step or total != self.total:
                        self.step, self.total = current, total
                        changed = True
            loss = LOSS_PATTERN.search(line)
            if loss:
                self.loss = float(loss.group(1))
                changed = True
            learning_rate = LR_PATTERN.search(line)
            if learning_rate:
                self.learning_rate = float(learning_rate.group(1))
                changed = True
        if not changed:
            return None
        metrics = {}
        if self.loss is not None:
            metrics["loss"] = self.loss
        if self.learning_rate is not None:
            metrics["learning_rate"] = self.learning_rate
        return {
            "phase": self.phase,
            "current": self.step,
            "total": self.total,
            "metrics": metrics,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
