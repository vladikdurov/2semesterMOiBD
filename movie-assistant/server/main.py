"""
HTTP-интерфейс рекомендательной системы (задание 4.1).
Реализован на стандартной библиотеке (http.server), без Flask.

Запуск:
    python server/main.py
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.engine import get_engine

engine = get_engine()
HOST = "127.0.0.1"
PORT = 5000


def _json_response(handler: BaseHTTPRequestHandler, payload, status: int = 200):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _error(handler: BaseHTTPRequestHandler, message: str, status: int = 400):
    _json_response(handler, {"error": message}, status)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)

        def arg(name, default=None):
            vals = qs.get(name)
            return vals[0] if vals else default

        def arg_int(name, default):
            raw = arg(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        try:
            if path in ("/", "/app", "/ui"):
                html_path = ROOT / "ui" / "index.html"
                body = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api":
                return _json_response(self, {
                    "service": "Movie Advisor API",
                    "docs": "/api/help",
                })

            if path == "/api/top":
                limit = max(1, min(arg_int("limit", 10), 50))
                return _json_response(self, engine.top_popular(limit))

            if path == "/api/by-genre":
                genre = (arg("genre") or "").strip()
                limit = max(1, min(arg_int("limit", 10), 50))
                try:
                    return _json_response(self, engine.top_by_genre(genre, limit))
                except ValueError as exc:
                    return _error(self, str(exc), 404)

            if path == "/api/similar-content":
                title = (arg("title") or "").strip()
                limit = max(1, min(arg_int("limit", 10), 50))
                try:
                    return _json_response(self, engine.similar_by_content(title, limit))
                except ValueError as exc:
                    return _error(self, str(exc), 404)

            if path == "/api/similar-collab":
                title = (arg("title") or "").strip()
                limit = max(1, min(arg_int("limit", 10), 50))
                try:
                    return _json_response(self, engine.similar_by_collab(title, limit))
                except ValueError as exc:
                    return _error(self, str(exc), 422)

            if path == "/api/genres":
                return _json_response(self, engine.available_genres())

            if path == "/api/help":
                return _json_response(self, [
                    {
                        "method": "GET",
                        "path": "/api/top",
                        "params": "limit=10",
                        "description": "Топ популярных фильмов по взвешенному рейтингу",
                    },
                    {
                        "method": "GET",
                        "path": "/api/by-genre",
                        "params": "genre=Comedy, limit=10",
                        "description": "Топ фильмов выбранного жанра",
                    },
                    {
                        "method": "GET",
                        "path": "/api/similar-content",
                        "params": "title=Matrix, limit=10",
                        "description": "10 похожих фильмов по контенту (жанры и год)",
                    },
                    {
                        "method": "GET",
                        "path": "/api/similar-collab",
                        "params": "title=Matrix, limit=10",
                        "description": "10 похожих фильмов по коллаборативной фильтрации",
                    },
                    {
                        "method": "GET",
                        "path": "/api/genres",
                        "params": "—",
                        "description": "Список жанров",
                    },
                    {
                        "method": "GET",
                        "path": "/api/help",
                        "params": "—",
                        "description": "Эта справка",
                    },
                ])

            _error(self, f"Неизвестный путь: {path}", 404)
        except Exception as exc:
            _error(self, f"Внутренняя ошибка: {exc}", 500)


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Movie Advisor API: http://{HOST}:{PORT}")
    print(f"Справка: http://{HOST}:{PORT}/api/help")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено")
        server.server_close()


if __name__ == "__main__":
    main()
