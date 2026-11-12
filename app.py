"""Tiny web UI (standard library only).

    .venv/bin/python app.py          # then open http://localhost:8000
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from emotion_detector.predictor import EmotionAnalyzer

PAGE = (Path(__file__).parent / "web" / "index.html").read_bytes()
analyzer = EmotionAnalyzer()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/analyze":
            return self._send(404, b"not found", "text/plain")
        try:
            length = min(int(self.headers.get("Content-Length", 0)), 20_000)
            req = json.loads(self.rfile.read(length) or b"{}")
            text = str(req.get("text", "")).strip()[:2000]
            if not text:
                return self._send(400, b'{"error": "empty text"}', "application/json")
            result = analyzer.analyze(text, str(req.get("context") or "")[:2000])
            self._send(200, json.dumps(result.to_dict()).encode(), "application/json")
        except (ValueError, json.JSONDecodeError):
            self._send(400, b'{"error": "bad request"}', "application/json")

    def log_message(self, *args):  # keep the console quiet
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Emotion detector running at http://localhost:8000  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
