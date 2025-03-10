from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import sqlite3
import bcrypt
import json
import datetime
from dotenv import load_dotenv
import os
from openai import OpenAI
import requests  # 添加requests库用于调用Brave API
import re

# 加载环境变量
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY')  # 从环境变量获取

# 添加Brave Search API密钥
BRAVE_API_KEY = os.environ.get('BRAVE_API_KEY')

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
        return User(user[0], user[1], user[2]) # user[2] 是 is_admin
    return None

def get_db_connection():
    conn = sqlite3.connect('chat_app.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
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
            # 修改：返回正确的 JSON 响应
            return jsonify({
                'success': True,
                'redirect': url_for('chat')
            })
        # 修改：返回错误的 JSON 响应
        return jsonify({
            'success': False,
            'error': '用户名或密码错误，请稍后重试！'
        }), 401  # 添加适当的状态码
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
            is_admin = 'is_admin' in request.form  # 复选框选中时为 True，否则 False
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                      (username, hashed_password, int(is_admin)))
        elif action == 'update':
            user_id = request.form['user_id']
            username = request.form['username']
            password = request.form['password']
            is_admin = 'is_admin' in request.form  # 复选框选中时为 True，否则 False
            if password:
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                c.execute('UPDATE users SET username = ?, password = ?, is_admin = ? WHERE id = ?',
                          (username, hashed_password, int(is_admin), user_id))
            else:
                c.execute('UPDATE users SET username = ?, is_admin = ? WHERE id = ?',
                          (username, int(is_admin), user_id))
        elif action == 'delete':
            user_id = request.form['user_id']
            c.execute('DELETE FROM users WHERE id = ? AND is_admin = 0', (user_id,))
            c.execute('DELETE FROM chat_history WHERE user_id = ?', (user_id,))
        conn.commit()
    
    # 修改查询以包含 is_admin 字段
    c.execute('''
        SELECT users.id, users.username, users.is_admin,
               COUNT(chat_history.id) as chat_count
        FROM users
        LEFT JOIN chat_history ON users.id = chat_history.user_id
        GROUP BY users.id, users.username, users.is_admin
    ''')
    users = [
        {
            'id': row['id'],
            'username': row['username'],
            'is_admin': bool(row['is_admin']),  # 确保转换为布尔值
            'chat_count': row['chat_count']
        }
        for row in c.fetchall()
    ]
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

@app.route('/chat/history')
@login_required
def get_chat_history():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT id, title FROM chat_history WHERE user_id = ? ORDER BY created_at DESC', 
              (current_user.id,))
    history = [{'id': row[0], 'title': row[1]} for row in c.fetchall()]
    conn.close()
    return jsonify(history)

@app.route('/chat/history/<int:chat_id>', methods=['GET', 'PUT', 'DELETE'])
@login_required
def manage_chat(chat_id):
    conn = get_db_connection()
    c = conn.cursor()
    
    # 验证权限
    c.execute('SELECT user_id FROM chat_history WHERE id = ?', (chat_id,))
    chat = c.fetchone()
    if not chat or chat[0] != current_user.id:
        conn.close()
        return jsonify({'error': 'Unauthorized'}), 403

    if request.method == 'GET':
        c.execute('SELECT content FROM chat_history WHERE id = ?', (chat_id,))
        content = c.fetchone()[0]
        messages = json.loads(content) if content else []
        conn.close()
        
        # 重置深度思考模式，加载聊天时回到默认状态
        if str(chat_id) in chat_model_settings:
            del chat_model_settings[str(chat_id)]
            
        return jsonify(messages)
    
    elif request.method == 'PUT':
        data = request.json
        c.execute('UPDATE chat_history SET title = ? WHERE id = ?', 
                  (data['title'], chat_id))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    
    elif request.method == 'DELETE':
        c.execute('DELETE FROM chat_history WHERE id = ?', (chat_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True})

@app.route('/chat/send', methods=['POST'])
@login_required
def send_message():
    # 删除此函数，因为已经有 handle_message 处理消息
    pass

def limit_context(messages, max_tokens=8192):  # 降低默认tokens数
    """限制上下文长度，避免超过API限制"""
    if not messages:
        return []
        
    total_tokens = 0
    limited_messages = []
    
    # 确保系统消息优先保留
    system_messages = [msg for msg in messages if msg['role'] == 'system']
    regular_messages = [msg for msg in messages if msg['role'] != 'system']
    
    # 估算系统消息的token数
    system_tokens = 0
    for msg in system_messages:
        estimated_tokens = len(msg['content']) // 3  # 粗略估计token数
        system_tokens += estimated_tokens
    
    # 保留最近的消息
    remaining_tokens = max_tokens - system_tokens
    
    for msg in reversed(regular_messages):
        estimated_tokens = len(msg['content']) // 3  # 更保守的token估计
        if total_tokens + estimated_tokens > remaining_tokens:
            break
        limited_messages.insert(0, msg)
        total_tokens += estimated_tokens
    
    # 添加回系统消息
    for msg in system_messages:
        limited_messages.insert(0, msg)
    
    print(f"原始消息数: {len(messages)}, 限制后消息数: {len(limited_messages)}, 估计token数: {total_tokens + system_tokens}")
    return limited_messages

def generate_title(first_message):
    """根据第一条消息生成标题"""
    if len(first_message) > 20:
        return first_message[:20] + "..."
    return first_message

# 初始化 Groq 客户端
client = OpenAI(
    base_url="https://careful-bat-89.deno.dev/api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# 保存每个聊天对话的模型设置
chat_model_settings = {}

# 新增调用Brave Search API的函数
def brave_web_search(query, count=5):
    """
    使用Brave Search API搜索网络内容
    
    参数:
    query -- 搜索查询字符串
    count -- 返回结果数量，默认为5
    
    返回:
    搜索结果列表
    """
    try:
        print(f"\n=== Brave搜索API调用 ===")
        print(f"搜索查询: {query}")
        print(f"请求结果数量: {count}")
        
        # 检查API密钥是否存在
        if not BRAVE_API_KEY:
            print(f"错误: 未设置Brave API密钥")
            return []
        
        # url = "https://api.search.brave.com/res/v1/web/search"
        url = "https://api.satelc.us.kg/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_API_KEY
        }
        params = {
            "q": query,
            "count": count,
            "search_lang": "zh-hans"  # 修正：使用 "zh-hans" 而不是 "zh"
        }
        
        print(f"API请求URL: {url}")
        print(f"请求参数: {params}")
        
        # 设置超时和重试
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
        except requests.exceptions.Timeout:
            print(f"API请求超时")
            return []
        except requests.exceptions.ConnectionError:
            print(f"API连接错误")
            return []
        
        print(f"API响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            # 添加数据结构验证
            if not isinstance(data, dict):
                print(f"错误: API返回的不是有效的JSON对象")
                return []
                
            if 'web' in data and 'results' in data['web']:
                for i, result in enumerate(data['web']['results']):
                    results.append({
                        'position': i + 1,
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'description': result.get('description', '')
                    })
            else:
                print(f"警告: API响应中未找到预期的结果结构")
                print(f"响应数据结构: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            
            print(f"搜索结果数量: {len(results)}")
            print(f"搜索结果摘要:")
            for i, res in enumerate(results):
                if i < 3:  # 只打印前3个结果摘要
                    print(f"  [{i+1}] {res['title'][:40]}... - {res['url']}")
                elif i == 3:
                    print(f"  ... 更多结果省略 ...")
            print("=========================\n")
            return results
        else:
            print(f"Brave Search API 错误: 状态码 {response.status_code}")
            print(f"错误详情: {response.text}")
            
            # 特殊处理422错误 - 参数问题
            if response.status_code == 422:
                try:
                    error_data = response.json()
                    if 'error' in error_data and 'meta' in error_data['error'] and 'errors' in error_data['error']['meta']:
                        for err in error_data['error']['meta']['errors']:
                            print(f"参数错误: {err.get('loc', [])} - {err.get('msg', '未知错误')}")
                except:
                    pass
            
            print("=========================\n")
            return []
            
    except Exception as e:
        print(f"调用Brave Search API时出错: {str(e)}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        print("=========================\n")
        return []

# 修改格式化搜索结果函数，添加target="_blank"
def format_search_results(results):
    """将搜索结果格式化为LLM可用的文本格式"""
    if not results:
        return "搜索未返回任何结果。"
    
    formatted_text = "以下是来自互联网的搜索结果:\n\n"
    
    for result in results:
        formatted_text += f"[{result['position']}] {result['title']}\n"
        # 添加target="_blank"属性，确保在新标签页打开链接
        formatted_text += f"URL: <a href=\"{result['url']}\" target=\"_blank\">{result['url']}</a>\n"
        formatted_text += f"描述: {result['description']}\n\n"
    
    print(f"\n=== 格式化的搜索结果 ===")
    print(f"结果条数: {len(results)}")
    print(f"格式化文本长度: {len(formatted_text)} 字符")
    print("=========================\n")
    
    return formatted_text

def send_message(message, history_id, deep_thinking=False, web_search=False):
    try:
        print(f"\n=== 消息处理开始 ===")
        print(f"用户消息: {message}")
        print(f"历史记录ID: {history_id}")
        print(f"深度思考模式: {deep_thinking}")
        print(f"网络搜索模式: {web_search}")
        
        # 从数据库获取历史消息
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT content FROM chat_history WHERE id = ? AND user_id = ?', (history_id, current_user.id))
        chat = c.fetchone()
        conn.close()
        if not chat:
            print(f"错误: 未找到聊天历史 ID {history_id}")
            return 'Chat not found', 404
        
        content = json.loads(chat['content'])
        print(f"历史消息数量: {len(content)}")
        
        # 准备消息列表
        messages = []
        
        # 根据是否开启了网络搜索来决定是否携带历史上下文
        if not web_search:
            # 未开启网络搜索时，携带历史上下文
            messages.extend(limit_context([{"role": msg["role"], "content": msg["content"]} for msg in content]))
            print(f"上下文限制后的消息数量: {len(messages)}")
        else:
            # 开启网络搜索时，不携带历史上下文，只处理当前用户消息
            print(f"网络搜索模式已开启，不携带历史上下文")
        
        # 如果开启了网络搜索，先进行搜索
        search_results = None
        if web_search:
            print(f"\n=== 开始网络搜索处理 ===")
            search_results = brave_web_search(message)
            
            if search_results:
                print(f"成功获取搜索结果，正在处理...")
                formatted_results = format_search_results(search_results)
                
                # 为大语言模型添加系统提示，指导如何引用搜索结果
                system_prompt = "用户开启了联网搜索功能，你将收到相关的搜索结果。请结合这些结果回答用户的问题。回答中需要引用相关内容的出处，按照以下格式引用:[数字] 文本内容，其中数字是搜索结果的编号。确保引用是相关的，并尽可能保持原文的准确性。在回答结束时，列出相关参考链接，格式为 '[数字] URL'。所有链接都应该在新标签页中打开。"
                print(f"添加系统提示: {system_prompt}")
                messages.insert(0, {
                    "role": "system", 
                    "content": system_prompt
                })
                
                # 在用户消息前添加搜索结果
                print(f"添加搜索结果到上下文")
                messages.append({"role": "system", "content": f"搜索结果：\n\n{formatted_results}"})
                print(f"添加搜索结果后的消息数量: {len(messages)}")
            else:
                print(f"搜索未返回结果或搜索失败")
                # 添加搜索失败的系统提示
                messages.insert(0, {
                    "role": "system", 
                    "content": "用户开启了联网搜索功能，但搜索未返回任何结果。请告知用户搜索未成功，并尽可能根据你的知识回答问题，同时说明信息可能不是最新的。"
                })
                print(f"添加搜索失败提示后的消息数量: {len(messages)}")

        # 添加新消息
        messages.append({"role": "user", "content": message})
        print(f"最终发送给LLM的消息数量: {len(messages)}")
        
        # 更新当前聊天的模型设置
        chat_model_settings[str(history_id)] = deep_thinking
        
        # 根据深度思考模式选择不同的模型和适当的max_tokens
        if deep_thinking:
            model = "deepseek-r1-distill-llama-70b"
            max_tokens = 8192  # r1模型的安全值
        else:
            model = "mixtral-8x7b-32768"
            max_tokens = 8192  # mixtral模型的安全值，确保小于8192
        
        print(f"\n=== 调用LLM API ===")
        print(f"使用模型: {model}")
        print(f"最大令牌数: {max_tokens}")
        
        # 使用 Groq API 发送请求
        response = client.chat.completions.create(
            model=model,  # 根据模式选择模型
            messages=messages,
            temperature=0.6,
            max_tokens=max_tokens  # 使用根据模型设置的安全值
        )

        # 获取回复内容
        assistant_message = response.choices[0].message.content
        
        # 添加打印系统回复
        print(f"\n=== LLM回复 ===")
        print(f"回复长度: {len(assistant_message)} 字符")
        print(f"回复摘要: {assistant_message[:100]}...")
        
        # 检查回复中是否包含引用
        if web_search and search_results:
            # 检查是否正确引用了搜索结果
            reference_pattern = r'\[\d+\]'
            references = re.findall(reference_pattern, assistant_message)
            print(f"检测到的引用数量: {len(references)}")
            
            # 检查是否包含参考链接部分
            if "参考链接" in assistant_message or "参考资料" in assistant_message:
                print(f"回复中包含参考链接部分")
            else:
                print(f"警告: 回复中可能缺少参考链接部分")
        
        print("=========================")

        # 保存对话到数据库
        conn = get_db_connection()
        c = conn.cursor()
        content.append({'role': 'user', 'content': message})
        content.append({'role': 'assistant', 'content': assistant_message})
        
        # 如果是新对话，生成标题
        if len(content) == 2:  # 第一轮对话
            title = generate_title(message[:20])
            c.execute('UPDATE chat_history SET title = ? WHERE id = ?', (title, history_id))
            
        c.execute('UPDATE chat_history SET content = ? WHERE id = ?', (json.dumps(content), history_id))
        conn.commit()
        conn.close()
        print(f"对话已保存到数据库")
        print(f"=== 消息处理完成 ===\n")

        return assistant_message

    except Exception as e:
        print(f"\n=== 处理消息时出错 ===")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        import traceback
        print(f"错误堆栈:\n{traceback.format_exc()}")
        print("=========================\n")
        raise

@app.route('/send_message', methods=['POST'])
def handle_message():
    try:
        data = request.get_json()
        message = data.get('message')
        history_id = data.get('history_id')
        deep_thinking = data.get('deep_thinking', False)  # 获取深度思考模式参数
        web_search = data.get('web_search', False)  # 获取网络搜索模式参数
        
        if not message:
            return jsonify({'error': '消息不能为空'}), 400
            
        response = send_message(message, history_id, deep_thinking, web_search)
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Error handling message: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/<int:user_id>/chats')
@login_required
def get_user_chats(user_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
        
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # 获取用户的所有聊天记录
        c.execute('''
            SELECT id, title, content, created_at 
            FROM chat_history 
            WHERE user_id = ? 
            ORDER BY created_at DESC
        ''', (user_id,))
        
        chats = []
        for row in c.fetchall():
            chat = {
                'id': row['id'],
                'title': row['title'],
                'created_at': row['created_at'],
                'messages': json.loads(row['content']) if row['content'] else []
            }
            chats.append(chat)
            
        conn.close()
        return jsonify(chats)
        
    except Exception as e:
        print(f"Error getting user chats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/<int:chat_id>', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
        
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # 验证聊天记录是否存在
        c.execute('SELECT id FROM chat_history WHERE id = ?', (chat_id,))
        if not c.fetchone():
            conn.close()
            return jsonify({'error': '聊天记录不存在'}), 404
            
        # 删除聊天记录
        c.execute('DELETE FROM chat_history WHERE id = ?', (chat_id,))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error deleting chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/<int:user_id>')
@login_required
def get_user_info(user_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
        
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT id, username, is_admin FROM users WHERE id = ?', (user_id,))
        user = c.fetchone()
        conn.close()
        
        if user:
            return jsonify({
                'id': user['id'],
                'username': user['username'],
                'is_admin': bool(user['is_admin'])
            })
        return jsonify({'error': '用户不存在'}), 404
        
    except Exception as e:
        print(f"Error getting user info: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()