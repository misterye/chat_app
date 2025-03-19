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
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import uuid
from werkzeug.utils import secure_filename

# 加载环境变量
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY')  # 从环境变量获取

# 添加Brave Search API密钥
BRAVE_API_KEY = os.environ.get('BRAVE_API_KEY')

# 添加 siliconflow 的 API 密钥
SILICONFLOW_API_KEY = os.environ.get('SILICONFLOW_API_KEY')

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
        url = "https://brave.binchat.top/res/v1/web/search"
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

def send_message(message, history_id, deep_thinking=False, web_search=False, doc_search=False):
    try:
        print(f"\n=== 消息处理开始 ===")
        print(f"用户消息: {message}")
        print(f"历史记录ID: {history_id}")
        print(f"深度思考模式: {deep_thinking}")
        print(f"网络搜索模式: {web_search}")
        print(f"文档检索模式: {doc_search}")
        
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

        # 如果启用了文档检索
        if doc_search:
            print(f"\n=== 开始文档检索处理 ===")
            
            # 获取用户的所有文档集合
            user_docs = session.get('user_docs', {})
            if not user_docs:
                # 如果session中没有，重新获取
                collections = []
                for item in os.listdir(VECTOR_DB_DIR):
                    if item.startswith('doc_'):
                        metadata_path = os.path.join(VECTOR_DB_DIR, item, 'metadata.pkl')
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'rb') as f:
                                metadata = pickle.load(f)
                                if metadata and len(metadata) > 0:
                                    first_meta = metadata[0]
                                    if first_meta.get('user_id') == current_user.id:
                                        collections.append(item)
                                        user_docs[item] = {
                                            'filename': first_meta.get('filename', '未知文件')
                                        }
                
                session['user_docs'] = user_docs
            
            if not user_docs:
                # 添加没有文档的提示
                messages.insert(0, {
                    "role": "system", 
                    "content": "用户请求检索文档，但当前没有上传任何文档。请告知用户需要先上传文档才能使用此功能。"
                })
            else:
                # 对所有文档集合进行检索
                all_results = []
                for collection_name in user_docs.keys():
                    try:
                        # 加载向量数据库
                        if collection_name not in vector_dbs:
                            vector_dbs[collection_name] = VectorDB.load(collection_name)
                            
                        vector_db = vector_dbs[collection_name]
                        
                        # 获取查询的嵌入向量
                        query_embedding = get_embeddings([message])
                        if query_embedding:
                            # 搜索相似文档
                            results = vector_db.search(query_embedding[0], k=3)  # 每个集合返回前3个结果
                            
                            for result in results:
                                result['collection'] = collection_name
                                result['filename'] = user_docs[collection_name].get('filename', '未知文件')
                                all_results.append(result)
                    except Exception as e:
                        print(f"检索文档 {collection_name} 时出错: {str(e)}")
                
                # 根据相似度得分排序所有结果
                all_results.sort(key=lambda x: x['score'])
                
                # 取前5个最相关的结果
                top_results = all_results[:5]
                
                if top_results:
                    # 格式化检索结果
                    formatted_results = "以下是来自您上传文档的相关内容:\n\n"
                    
                    for i, result in enumerate(top_results):
                        formatted_results += f"[文档 {i+1}] {result['filename']}\n"
                        formatted_results += f"内容: {result['text'][:300]}...\n\n"
                    
                    # 添加系统提示
                    system_prompt = "用户已启用文档检索功能，下面是检索到的相关文档内容。请使用这些信息回答用户的问题，引用相关内容时请标注来自哪个文档。"
                    messages.insert(0, {
                        "role": "system", 
                        "content": system_prompt
                    })
                    
                    # 添加检索结果到上下文
                    messages.append({
                        "role": "system", 
                        "content": f"检索结果：\n\n{formatted_results}"
                    })
                else:
                    # 没有找到相关内容
                    messages.insert(0, {
                        "role": "system", 
                        "content": "用户请求检索文档内容，但未找到与查询相关的内容。请告知用户未找到相关信息，并询问是否需要更改搜索词或上传更多相关文档。"
                    })
        
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
        doc_search = data.get('doc_search', False)  # 获取文档检索模式参数
        
        if not message:
            return jsonify({'error': '消息不能为空'}), 400
            
        response = send_message(message, history_id, deep_thinking, web_search, doc_search)
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

# 创建向量存储目录
VECTOR_DB_DIR = 'vector_db'
if not os.path.exists(VECTOR_DB_DIR):
    os.makedirs(VECTOR_DB_DIR)

# 向量数据库相关配置
class VectorDB:
    def __init__(self, dimension=1024):  # 修改默认维度为1024，匹配BGE模型
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # 使用L2距离的FAISS索引
        self.texts = []  # 存储文本片段
        self.metadata = []  # 存储元数据
        self.collection_info = {}  # 存储集合信息

    def add_texts(self, texts, metadata=None):
        """添加文本向量到数据库"""
        if not texts:
            return []
            
        ids = []
        for i, text in enumerate(texts):
            text_id = len(self.texts)
            self.texts.append(text)
            
            meta = metadata[i] if metadata and i < len(metadata) else {}
            self.metadata.append(meta)
            
            ids.append(text_id)
        
        return ids
    
    def add_vectors(self, vectors, texts, metadata=None):
        """添加向量到数据库"""
        if len(vectors) == 0:
            return []
            
        # 转换向量为numpy数组
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        
        # 如果是单个向量，添加一个维度
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
            
        # 确保向量维度正确
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"向量维度 {vectors.shape[1]} 与索引维度 {self.dimension} 不匹配")
            
        # 添加到FAISS索引
        ids = self.add_texts(texts, metadata)
        faiss.normalize_L2(vectors)  # 归一化向量
        self.index.add(vectors)
        
        return ids
    
    def search(self, query_vector, k=5):
        """搜索最相似的文档"""
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
            
        # 确保向量是二维的
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        faiss.normalize_L2(query_vector)  # 归一化查询向量
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts) and idx >= 0:
                results.append({
                    'id': int(idx),
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'score': float(distances[0][i])
                })
        
        return results
    
    def save(self, collection_name):
        """保存索引和元数据到磁盘"""
        collection_path = os.path.join(VECTOR_DB_DIR, collection_name)
        if not os.path.exists(collection_path):
            os.makedirs(collection_path)
            
        # 保存FAISS索引
        index_path = os.path.join(collection_path, 'index.faiss')
        faiss.write_index(self.index, index_path)
        
        # 保存文本和元数据
        texts_path = os.path.join(collection_path, 'texts.pkl')
        metadata_path = os.path.join(collection_path, 'metadata.pkl')
        
        with open(texts_path, 'wb') as f:
            pickle.dump(self.texts, f)
            
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        # 更新集合信息
        info = {
            'dimension': self.dimension,
            'count': len(self.texts),
            'created_at': datetime.datetime.now().isoformat()
        }
        self.collection_info = info
        
        info_path = os.path.join(collection_path, 'info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f)
            
        return collection_path
    
    @classmethod
    def load(cls, collection_name):
        """从磁盘加载索引和元数据"""
        collection_path = os.path.join(VECTOR_DB_DIR, collection_name)
        if not os.path.exists(collection_path):
            raise ValueError(f"集合 {collection_name} 不存在")
        
        # 读取信息（先获取维度信息）
        info_path = os.path.join(collection_path, 'info.json')
        with open(info_path, 'r') as f:
            info = json.load(f)
            
        # 使用正确的维度创建实例
        dimension = info.get('dimension', 1024)  # 如果没有维度信息，默认使用1024
        instance = cls(dimension=dimension)
        
        # 加载索引
        index_path = os.path.join(collection_path, 'index.faiss')
        instance.index = faiss.read_index(index_path)
        
        # 加载文本和元数据
        texts_path = os.path.join(collection_path, 'texts.pkl')
        metadata_path = os.path.join(collection_path, 'metadata.pkl')
        
        with open(texts_path, 'rb') as f:
            instance.texts = pickle.load(f)
            
        with open(metadata_path, 'rb') as f:
            instance.metadata = pickle.load(f)
        
        # 设置集合信息
        instance.collection_info = info
        
        return instance

# 全局变量存储加载的向量数据库
vector_dbs = {}

# 文本分段函数
def split_text(text, method='delimiter', **kwargs):
    """
    将文本分段
    
    参数:
    text -- 要分段的文本
    method -- 分段方法，可选 'delimiter', 'chunk_size', 'semantic'
    kwargs -- 其他参数，例如 delimiter, chunk_size 等
    
    返回:
    分段后的文本列表
    """
    if method == 'delimiter':
        delimiter = kwargs.get('delimiter', '####')
        chunks = [chunk.strip() for chunk in text.split(delimiter) if chunk.strip()]
        return chunks
        
    elif method == 'chunk_size':
        chunk_size = kwargs.get('chunk_size', 1000)
        overlap = kwargs.get('overlap', 0)
        
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            if i + chunk_size <= len(text):
                chunks.append(text[i:i + chunk_size])
            else:
                chunks.append(text[i:])
                break
        return chunks
        
    elif method == 'semantic':
        # 简单的语义分段，按句子分割然后合并
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        chunk_size = kwargs.get('chunk_size', 5)  # 每个块中的句子数
        
        if len(sentences) <= chunk_size:
            return [text]
            
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
        
    else:
        # 默认返回原文本
        return [text]

# 获取嵌入向量
def get_embeddings(texts):
    """
    获取文本的嵌入向量，使用 siliconflow 的 API，处理token限制
    
    参数:
    texts -- 文本列表
    
    返回:
    嵌入向量列表
    """
    if not texts:
        return []
        
    try:
        print(f"\n=== 获取文本嵌入向量 ===")
        print(f"文本数量: {len(texts)}")
        
        # 检查 API 密钥是否存在
        if not SILICONFLOW_API_KEY:
            print(f"错误: 未设置 SILICONFLOW_API_KEY")
            return []
            
        url = "https://api.siliconflow.cn/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }
        
        embeddings = []
        
        # 估算token长度的函数（简单实现）
        def estimate_tokens(text):
            # 粗略估计：中文每字约1个token，英文每4个字符约1个token
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            other_chars = len(text) - chinese_chars
            return chinese_chars + (other_chars // 4)
        
        # API的token上限
        TOKEN_LIMIT = 500  # 设置为500而不是512，留一些余量
        
        for i, text in enumerate(texts):
            if i < 5:  # 只打印前5个文本示例
                print(f"获取嵌入向量: 文本[{i+1}]: {text[:50]}...")
                
            # 估计token数量
            estimated_tokens = estimate_tokens(text)
            
            # 如果超过限制，截断文本
            if estimated_tokens > TOKEN_LIMIT:
                print(f"警告: 文本片段[{i+1}]超过token限制，将被截断。估计tokens: {estimated_tokens}")
                # 按比例截断
                truncate_ratio = TOKEN_LIMIT / estimated_tokens
                truncate_length = int(len(text) * truncate_ratio)
                truncated_text = text[:truncate_length]
                print(f"  截断后长度: {len(truncated_text)} 字符")
                text = truncated_text
            
            payload = {
                "model": "BAAI/bge-large-zh-v1.5",
                "input": text,
                "encoding_format": "float"
            }
            
            # 尝试API调用，如果失败则继续截断
            max_retries = 3
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    response = requests.post(url, json=payload, headers=headers)
                    
                    if response.status_code == 200:
                        result = response.json()
                        # 提取嵌入向量
                        if 'data' in result and len(result['data']) > 0 and 'embedding' in result['data'][0]:
                            embedding = result['data'][0]['embedding']
                            embeddings.append(embedding)
                            success = True
                        else:
                            print(f"警告: API 返回中未找到嵌入向量，返回内容: {result}")
                            if retry_count == max_retries - 1:
                                return []
                    else:
                        error_msg = f"错误: API 请求失败，状态码: {response.status_code}"
                        
                        # 检查是否是token限制问题
                        if 'input must have less than 512 tokens' in response.text:
                            # 继续截断文本
                            text = text[:int(len(text) * 0.7)]  # 截断到原长度的70%
                            payload["input"] = text
                            print(f"  Token限制错误，进一步截断文本至 {len(text)} 字符")
                        else:
                            print(f"{error_msg}, 返回内容: {response.text}")
                            if retry_count == max_retries - 1:
                                return []
                    
                    retry_count += 1
                    
                except Exception as e:
                    print(f"API调用异常: {str(e)}")
                    retry_count += 1
                    if retry_count == max_retries:
                        return []
            
        print(f"成功获取 {len(embeddings)} 个嵌入向量")
        print("=========================\n")
        
        return embeddings
    except Exception as e:
        print(f"获取嵌入向量时出错: {str(e)}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        return []

# 创建docs目录存储上传的文档
DOCS_DIR = 'docs'
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

# 处理文件上传并向量化
@app.route('/api/upload_document', methods=['POST'])
@login_required
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
            
        # 验证文件格式
        if not file.filename.endswith('.txt'):
            return jsonify({'error': '目前只支持上传txt格式文件'}), 400
            
        # 生成安全的文件名
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(DOCS_DIR, filename)
        
        # 保存文件
        file.save(filepath)
        
        # 读取文件内容
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 如果文件内容为空
        if not text.strip():
            return jsonify({'error': '文件内容为空'}), 400
            
        # 使用文件名作为集合名称的一部分
        collection_name = f"doc_{filename.split('.')[0]}"
        
        # 分段并向量化
        split_method = request.form.get('split_method', 'chunk_size')
        
        # 修改默认分段大小，确保不超过token限制
        # 一般来说，1个中文字符约为1个token，所以设置为400字符较为安全
        chunk_size = int(request.form.get('chunk_size', 400))
        overlap = int(request.form.get('overlap', 50))
        
        split_params = {
            'chunk_size': chunk_size,
            'overlap': overlap
        }
        
        # 生成分段
        chunks = split_text(text, method=split_method, **split_params)
        
        if not chunks:
            return jsonify({'error': '文本分段失败，未生成任何文本块'}), 400
            
        # 获取嵌入向量
        embeddings = get_embeddings(chunks)
        
        if not embeddings or len(embeddings) != len(chunks):
            return jsonify({'error': f'嵌入向量生成失败: 预期{len(chunks)}个向量，得到{len(embeddings)}个'}), 500
            
        # 创建向量数据库
        vector_db = VectorDB()
        
        # 生成元数据
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                'chunk_id': i,
                'filename': file.filename,
                'filepath': filepath,
                'user_id': current_user.id,
                'username': current_user.username,
                'created_at': datetime.datetime.now().isoformat(),
                'chars': len(chunk),
                'tokens': len(chunk) // 4  # 粗略估计token数量
            })
            
        # 添加向量
        vector_db.add_vectors(embeddings, chunks, metadata)
        
        # 保存到磁盘
        collection_path = vector_db.save(collection_name)
        
        # 添加到内存中的数据库列表
        vector_dbs[collection_name] = vector_db
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'collection_name': collection_name,
            'chunks': len(chunks),
            'path': collection_path,
            'info': vector_db.collection_info
        })
        
    except Exception as e:
        print(f"文件上传处理错误: {str(e)}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# 获取用户上传的文档列表
@app.route('/api/user_documents', methods=['GET'])
@login_required
def get_user_documents():
    try:
        collections = []
        docs_info = {}
        
        # 扫描向量数据库目录
        for item in os.listdir(VECTOR_DB_DIR):
            if item.startswith('doc_'):
                info_path = os.path.join(VECTOR_DB_DIR, item, 'info.json')
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    
                    # 加载第一个chunk的元数据以获取文件信息
                    metadata_path = os.path.join(VECTOR_DB_DIR, item, 'metadata.pkl')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'rb') as f:
                            metadata = pickle.load(f)
                            if metadata and len(metadata) > 0:
                                first_meta = metadata[0]
                                if first_meta.get('user_id') == current_user.id:
                                    collections.append({
                                        'collection_name': item,
                                        'filename': first_meta.get('filename', '未知文件'),
                                        'count': info.get('count', 0),
                                        'created_at': info.get('created_at', '')
                                    })
                                    
                                    # 保存文档信息用于聊天时显示
                                    docs_info[item] = {
                                        'filename': first_meta.get('filename', '未知文件')
                                    }
        
        # 保存到session中以便聊天时使用
        session['user_docs'] = docs_info
        
        return jsonify(collections)
                
    except Exception as e:
        print(f"获取用户文档列表时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()