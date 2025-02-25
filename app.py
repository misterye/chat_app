from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import sqlite3
import bcrypt
import json
import datetime
from dotenv import load_dotenv
import os
from openai import OpenAI

# 加载环境变量
load_dotenv()

app = Flask(__name__)
app.secret_key = 'e3bcf3078592b37dab1a1ad1264707e1fd7030d655528cd7a1ad202edaf1a5db'  # 请替换为安全的密钥

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, is_admin):
        self.id = id
        self.username = username
        self.is_admin = is_admin

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('chat_app.db')
    c = conn.cursor()
    c.execute('SELECT id, username, is_admin FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    conn.close()
    if user:
        return User(user[0], user[1], user[2])
    return None

def get_db_connection():
    conn = sqlite3.connect('chat_app.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            user_obj = User(user['id'], user['username'], user['is_admin'])
            login_user(user_obj)
            return redirect(url_for('chat'))
        return 'Invalid username or password'
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    if not current_user.is_admin:
        return 'Access denied'
    conn = get_db_connection()
    c = conn.cursor()
    if request.method == 'POST':
        action = request.form['action']
        if action == 'add':
            username = request.form['username']
            password = request.form['password']
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        elif action == 'delete':
            user_id = request.form['user_id']
            c.execute('DELETE FROM users WHERE id = ?', (user_id,))
            c.execute('DELETE FROM chat_history WHERE user_id = ?', (user_id,))
        elif action == 'update':
            user_id = request.form['user_id']
            username = request.form['username']
            password = request.form['password']
            if password:
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                c.execute('UPDATE users SET username = ?, password = ? WHERE id = ?', (username, hashed_password, user_id))
            else:
                c.execute('UPDATE users SET username = ? WHERE id = ?', (username, user_id))
        conn.commit()
    c.execute('SELECT id, username FROM users')
    users = c.fetchall()
    conn.close()
    return render_template('admin.html', users=users)

@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html')

@app.route('/chat/new', methods=['POST'])
@login_required
def new_chat():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('INSERT INTO chat_history (user_id, title, content) VALUES (?, ?, ?)',
              (current_user.id, '新聊天', json.dumps([])))
    chat_id = c.lastrowid
    conn.commit()
    conn.close()
    return jsonify({'chat_id': chat_id})

@app.route('/chat/history', methods=['GET'])
@login_required
def get_history():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT id, title, created_at FROM chat_history WHERE user_id = ? ORDER BY created_at DESC',
              (current_user.id,))
    history = c.fetchall()
    conn.close()
    return jsonify([{'id': row['id'], 'title': row['title'], 'created_at': row['created_at']} for row in history])

@app.route('/chat/history/<int:history_id>', methods=['GET', 'PUT', 'DELETE'])
@login_required
def manage_history(history_id):
    conn = get_db_connection()
    c = conn.cursor()
    if request.method == 'GET':
        c.execute('SELECT content FROM chat_history WHERE id = ? AND user_id = ?', (history_id, current_user.id))
        chat = c.fetchone()
        if chat:
            return jsonify(json.loads(chat['content']))
        return 'Chat not found', 404
    elif request.method == 'PUT':
        title = request.json['title']
        c.execute('UPDATE chat_history SET title = ? WHERE id = ? AND user_id = ?',
                  (title, history_id, current_user.id))
        conn.commit()
        return 'OK'
    elif request.method == 'DELETE':
        c.execute('DELETE FROM chat_history WHERE id = ? AND user_id = ?', (history_id, current_user.id))
        conn.commit()
        return 'OK'
    conn.close()

# 初始化 Groq 客户端
client = OpenAI(
    base_url="https://careful-bat-89.deno.dev/api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

def send_message(message, history_id):
    try:
        # 从数据库获取历史消息
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT content FROM chat_history WHERE id = ? AND user_id = ?', (history_id, current_user.id))
        chat = c.fetchone()
        conn.close()
        if not chat:
            return 'Chat not found', 404
        
        content = json.loads(chat['content'])
        
        # 准备消息列表
        messages = [
            {"role": "system", "content": "你是一个有帮助的助手。"},
        ]
        # 添加历史消息
        messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in content])
        # 添加新消息
        messages.append({"role": "user", "content": message})

        # 使用 Groq API 发送请求
        response = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",  # 或其他支持的模型
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )

        # 获取回复内容
        assistant_message = response.choices[0].message.content

        # 保存对话到数据库
        conn = get_db_connection()
        c = conn.cursor()
        content.append({'role': 'user', 'content': message})
        content.append({'role': 'assistant', 'content': assistant_message})
        c.execute('UPDATE chat_history SET content = ? WHERE id = ?', (json.dumps(content), history_id))
        conn.commit()
        conn.close()

        return assistant_message

    except Exception as e:
        print(f"Error in send_message: {str(e)}")
        raise

@app.route('/send_message', methods=['POST'])
def handle_message():
    try:
        data = request.get_json()
        message = data.get('message')
        history_id = data.get('history_id')
        
        if not message:
            return jsonify({'error': '消息不能为空'}), 400
            
        response = send_message(message, history_id)
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Error handling message: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)