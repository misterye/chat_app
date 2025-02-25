import sqlite3
import bcrypt

conn = sqlite3.connect('chat_app.db')
c = conn.cursor()

# 创建用户表
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    is_admin INTEGER DEFAULT 0
)
''')

# 创建聊天历史记录表
c.execute('''
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
)
''')

# 添加默认管理员用户
admin_username = 'admin'
admin_password = 'admin'
hashed_password = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt())
c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
          (admin_username, hashed_password, 1))

conn.commit()
conn.close()