# Gunicorn 配置
workers = 4  # 建议设置为 CPU 核心数 * 2 + 1
bind = "127.0.0.1:8503"  # 内部端口，将由 Nginx 代理
timeout = 120
keepalive = 5
worker_class = "sync"
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log" 