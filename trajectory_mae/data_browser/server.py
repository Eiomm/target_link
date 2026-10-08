"""Loopback-only read-only HTTP service for the trajectory explorer."""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from trajectory_mae.data_browser.data import Explorer

STATIC = Path(__file__).parent / 'static'


def is_loopback_host(host):
    """Validate the browser-facing host; forwarding may change its port."""
    match = re.fullmatch(r'(localhost|127\.0\.0\.1|\[::1\])(?::([0-9]{1,5}))?', host, re.IGNORECASE)
    return bool(match and (match[2] is None or 1 <= int(match[2]) <= 65535))


def handler_for(explorer):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, kind='application/json; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'self'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            # Opening a link from another page is a legitimate top-level navigation.
            # Keep cross-site data requests and embedded pages blocked.
            host = self.headers.get('Host', '')
            if len(self.headers.get_all('Host', [])) != 1 or not is_loopback_host(host):
                self.send(403, b'{"error":"Loopback host required"}')
                return
            url = urlparse(self.path)
            homepage_navigation = (
                url.path in ('/', '/index.html')
                and self.headers.get('Sec-Fetch-Mode') == 'navigate'
                and self.headers.get('Sec-Fetch-Dest') == 'document'
            )
            if self.headers.get('Sec-Fetch-Site') == 'cross-site' and not homepage_navigation:
                self.send(403, b'{"error":"Same-origin requests required"}')
                return
            try:
                args = {k:v[-1] for k,v in parse_qs(url.query, max_num_fields=20).items()}
                routes = {'/api/overview': explorer.overview, '/api/cells': explorer.cells,
                          '/api/locate': explorer.locate, '/api/cell': explorer.cell, '/api/group': explorer.group}
                if url.path == '/api/health':
                    result = {'status':'ready'}
                elif url.path in routes:
                    result = routes[url.path](**args)
                elif url.path in ('/','/index.html','/app.js','/style.css','/favicon.svg'):
                    path = STATIC / ('index.html' if url.path == '/' else url.path[1:])
                    self.send(200, path.read_bytes(), (mimetypes.guess_type(path)[0] or 'application/octet-stream')+'; charset=utf-8')
                    return
                else:
                    raise LookupError('页面不存在')
                self.send(200, json.dumps(result, ensure_ascii=False, allow_nan=False).encode())
            except (ValueError, TypeError) as e:
                self.send(400, json.dumps({'error':str(e)},ensure_ascii=False).encode())
            except LookupError as e:
                self.send(404, json.dumps({'error':str(e)},ensure_ascii=False).encode())
            except Exception:
                traceback.print_exc()
                self.send(500, json.dumps({'error':'读取失败，请检查本地服务日志；原始数据未修改。'},ensure_ascii=False).encode())
    return Handler


def main():
    p = argparse.ArgumentParser(description='本地轨迹数据浏览器（只读）')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--repo', type=Path, default=ROOT)
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    if not 1024<=a.port<=65535:
        p.error('port must be in 1024..65535')
    explorer = Explorer(a.repo, a.report)
    server = ThreadingHTTPServer(('127.0.0.1', a.port), handler_for(explorer))
    server.daemon_threads=True
    print(f'轨迹数据浏览器已启动：http://127.0.0.1:{a.port}',flush=True)
    print('仅本机可访问；原始数据与统计报告只读。',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__=='__main__':
    main()
