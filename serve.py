import sys
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        web_app_dir = os.path.join(os.path.dirname(__file__), "web_app")
        super().__init__(*args, directory=web_app_dir, **kwargs)

    def log_message(self, format, *args):
        # Concise logging
        sys.stdout.write("%s - - [%s] %s\n" %
                         (self.address_string(),
                          self.log_date_time_string(),
                          format%args))

if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    port = 8000
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, CustomHandler)
    print(f"[*] Multi-Threaded HTTP Server running at http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down.")
        httpd.server_close()
