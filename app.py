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

# 设置文档和向量数据库路径
DOCS_DIR = os.path.join(os.getcwd(), 'docs')
VECTOR_DB_DIR = os.path.join(os.getcwd(), 'vector_db')

# 确保文档和向量数据库目录存在
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# 定义文档分类目录
DOC_CATEGORIES = ['clients', 'companies', 'products', 'techdocs']

# 确保每个分类目录都存在
for category in DOC_CATEGORIES:
    os.makedirs(os.path.join(DOCS_DIR, category), exist_ok=True)
    os.makedirs(os.path.join(VECTOR_DB_DIR, category), exist_ok=True)

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

# 获取指定分类下的文档
@app.route('/api/category_documents/<category>', methods=['GET'])
@login_required
def get_category_documents(category):
    try:
        # 验证分类是否有效
        if category not in DOC_CATEGORIES:
            return jsonify({'error': '无效的文档分类'}), 400
            
        category_path = os.path.join(DOCS_DIR, category)
        category_vector_path = os.path.join(VECTOR_DB_DIR, category)
        
        # 检查分类目录是否存在
        if not os.path.exists(category_path):
            os.makedirs(category_path, exist_ok=True)
            
        if not os.path.exists(category_vector_path):
            os.makedirs(category_vector_path, exist_ok=True)
            
        # 获取该分类下的所有文档
        documents = []
        has_documents = False
        
        # 检查原始文档目录
        if os.path.exists(category_path) and os.listdir(category_path):
            has_documents = True
            
        # 检查向量数据库目录 - 使用优化的方法避免加载大型pickle文件
        if os.path.exists(category_vector_path):
            for item in os.listdir(category_vector_path):
                item_path = os.path.join(category_vector_path, item)
                info_path = os.path.join(item_path, 'info.json')
                
                # 优先检查info文件（小文件）
                if os.path.isdir(item_path) and os.path.exists(info_path):
                    try:
                        # 读取信息文件
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                        
                        # 检查用户ID - 首先从info文件中获取
                        user_id = None
                        filename = '未知文件'
                        
                        # 从info直接检查用户ID (优先)
                        if 'user_id' in info:
                            user_id = info.get('user_id')
                            filename = info.get('filename', '未知文件')
                        # 从metadata_sample检查
                        elif 'metadata_sample' in info and 'user_id' in info.get('metadata_sample', {}):
                            user_id = info['metadata_sample'].get('user_id')
                            filename = info['metadata_sample'].get('filename', '未知文件')
                        else:
                            # 仅在必要时回退到检查metadata文件
                            metadata_path = os.path.join(item_path, 'metadata.pkl')
                            if os.path.exists(metadata_path):
                                try:
                                    # 只读取第一条记录
                                    with open(metadata_path, 'rb') as f:
                                        metadata_start = pickle.load(f)
                                        if metadata_start and len(metadata_start) > 0:
                                            first_meta = metadata_start[0]
                                            user_id = first_meta.get('user_id')
                                            filename = first_meta.get('filename', '未知文件')
                                except Exception as e:
                                    print(f"读取元数据时出错: {str(e)}")
                        
                        # 检查是否属于当前用户
                        if user_id is not None and user_id == current_user.id:
                            documents.append({
                                'collection_name': item,
                                'filename': filename,
                                'count': info.get('count', 0),
                                'created_at': info.get('created_at', '')
                            })
                            has_documents = True
                    except Exception as e:
                        print(f"读取文档信息时出错: {str(e)}")
        
        return jsonify({
            'category': category,
            'has_documents': has_documents,
            'documents': documents
        })
    except Exception as e:
        print(f"获取分类文档时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
                except Exception:
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

# 向量数据库类
class VectorDB:
    def __init__(self, collection_name, category, embedding_model=None):
        self.collection_name = collection_name
        self.category = category
        try:
            self.embedding_model = embedding_model or get_embedding_model()  # 使用全局单例
        except Exception as e:
            print(f"初始化嵌入模型时出错: {str(e)}")
            self.embedding_model = None
        self.base_path = os.path.join(VECTOR_DB_DIR, category, collection_name)
        
        # 创建必要的目录
        os.makedirs(self.base_path, exist_ok=True)
        
        # 状态变量，跟踪是否已有索引
        self.index_exists = False
        self.dimension = None
        self.texts = []
        self.metadata = []
        
        # 检查是否存在已保存的索引
        index_path = os.path.join(self.base_path, 'index.faiss')
        texts_path = os.path.join(self.base_path, 'texts.pkl')
        metadata_path = os.path.join(self.base_path, 'metadata.pkl')
        
        if os.path.exists(index_path) and os.path.exists(texts_path) and os.path.exists(metadata_path):
            try:
                # 已有索引，加载现有数据
                self.index_exists = True
                
                # 加载文本和元数据
                with open(texts_path, 'rb') as f:
                    self.texts = pickle.load(f)
                    
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                    
                # 读取索引以获取维度信息
                temp_index = faiss.read_index(index_path)
                self.dimension = temp_index.d
                del temp_index  # 释放内存
            except Exception as e:
                print(f"加载现有向量数据库时出错: {str(e)}")
                # 重置状态
                self.index_exists = False
                self.dimension = None
                self.texts = []
                self.metadata = []
    
    def add_documents(self, texts, metadata=None, batch_size=16):
        """添加文档到向量数据库 - 支持增量添加"""
        import gc  # 垃圾回收
        
        if not texts:
            return
            
        metadata = metadata or [{}] * len(texts)
        
        # 首先记录当前文本和元数据
        start_idx = len(self.texts)
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        
        try:
            # 获取文档的嵌入向量（批处理方式）
            embeddings = get_embeddings(texts, batch_size=batch_size)
            if embeddings is None:
                raise Exception("获取嵌入向量失败")
        except Exception as e:
            print(f"获取嵌入向量时出错: {str(e)}")
            # 清理已添加的文本和元数据
            self.texts = self.texts[:start_idx]
            self.metadata = self.metadata[:start_idx]
            raise Exception(f"无法加载任何嵌入模型，请检查网络连接或手动下载模型。详细错误: {str(e)}")
        
        if not self.index_exists:
            # 首次创建索引
            self.dimension = embeddings.shape[1]
            
            # 使用内存更高效的索引类型
            # 数据量少时使用简单的IndexFlatL2，数据量大时使用IVFFlat
            if len(texts) > 1000:
                # 使用IVFFlat更节省内存
                nlist = min(64, len(texts))  # 聚类中心数量
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
                
                # 必须先训练再添加
                index.train(embeddings.astype(np.float32))
            else:
                # 数据量小时用基本索引
                index = faiss.IndexFlatL2(self.dimension)
                
            # 添加向量
            index.add(embeddings.astype(np.float32))
            self.index_exists = True
        else:
            # 增量更新已有索引
            index = faiss.read_index(os.path.join(self.base_path, 'index.faiss'))
            
            # 检查维度是否匹配
            if index.d != embeddings.shape[1]:
                raise ValueError(f"嵌入向量维度不匹配: 预期 {index.d}, 得到 {embeddings.shape[1]}")
            
            # 添加新向量
            index.add(embeddings.astype(np.float32))
        
        # 保存索引、文本和元数据
        faiss.write_index(index, os.path.join(self.base_path, 'index.faiss'))
        
        # 释放索引内存
        del index
        
        # 保存全部文本和元数据
        with open(os.path.join(self.base_path, 'texts.pkl'), 'wb') as f:
            pickle.dump(self.texts, f)
            
        with open(os.path.join(self.base_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # 保存索引信息
        info = {
            'count': len(self.texts),
            'dimension': self.dimension,
            'created_at': datetime.datetime.now().isoformat(),
            'last_updated': datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(self.base_path, 'info.json'), 'w') as f:
            json.dump(info, f)
            
        # 强制垃圾回收
        gc.collect()
    
    def search(self, query_embedding, k=5):
        """搜索最相似的文档"""
        # 加载索引
        index_path = os.path.join(self.base_path, 'index.faiss')
        if not os.path.exists(index_path):
            return []
            
        index = faiss.read_index(index_path)
        
        # 加载文本和元数据
        with open(os.path.join(self.base_path, 'texts.pkl'), 'rb') as f:
            texts = pickle.load(f)
            
        with open(os.path.join(self.base_path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # 搜索
        query_embedding = np.array([query_embedding]).astype(np.float32)
        scores, indices = index.search(query_embedding, k)
        
        # 构建结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(texts):
                results.append({
                    'text': texts[idx],
                    'metadata': metadata[idx],
                    'score': float(scores[0][i])
                })
        
        return results
    
    @classmethod
    def load(cls, collection_name, category):
        """加载向量数据库"""
        base_path = os.path.join(VECTOR_DB_DIR, category, collection_name)
        
        if not os.path.exists(base_path) or not os.path.exists(os.path.join(base_path, 'index.faiss')):
            raise ValueError(f"向量数据库 {collection_name} 在类别 {category} 下不存在")
        
        instance = cls(collection_name, category)
        return instance

# 全局向量数据库缓存
vector_dbs = {}

# 全局变量，用于存储模型实例
embedding_model = None

def get_embedding_model():
    """获取或初始化模型实例（单例模式）- 使用SiliconFlow API"""
    # 这个函数现在只是一个占位符，实际上我们不再需要加载本地模型
    # 而是通过API调用SiliconFlow的嵌入服务
    print("使用SiliconFlow API进行嵌入向量生成")
    
    # 检查API密钥是否存在
    if not SILICONFLOW_API_KEY:
        print("警告: 未设置SiliconFlow API密钥，嵌入功能可能无法正常工作")
    
    return None

# 获取嵌入向量函数
def get_embeddings(texts, batch_size=32):
    """获取文本的嵌入向量，使用SiliconFlow API"""
    try:
        # 检查API密钥是否存在
        if not SILICONFLOW_API_KEY:
            print("错误: 未设置SiliconFlow API密钥")
            # 返回一个空的嵌入向量而不是抛出异常
            print("返回空向量作为替代")
            # 创建一个与预期维度相同的空向量数组
            return np.zeros((len(texts), 1024))
        
        print(f"使用SiliconFlow API处理{len(texts)}个文本")
        
        # API端点
        url = "https://api.siliconflow.cn/v1/embeddings"
        
        # 处理单个文本或文本列表
        if len(texts) == 1:
            # 单个文本处理
            payload = {
                "model": "BAAI/bge-large-zh-v1.5",
                "input": texts[0],
                "encoding_format": "float"
            }
            
            headers = {
                "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if "data" in result and len(result["data"]) > 0 and "embedding" in result["data"][0]:
                    # 返回单个嵌入向量
                    embedding = np.array(result["data"][0]["embedding"])
                    return np.array([embedding])  # 保持与批处理返回格式一致
                else:
                    print(f"API响应格式错误: {result}")
                    raise Exception("API响应格式错误")
            else:
                print(f"API请求失败: {response.status_code}, {response.text}")
                raise Exception(f"API请求失败: {response.status_code}")
        else:
            # 批处理文本
            all_embeddings = []
            
            # 分批处理，每次处理batch_size个文本
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                print(f"处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}, 大小: {len(batch)}")
                
                # 对于批处理，我们需要多次调用API
                batch_embeddings = []
                for text in batch:
                    payload = {
                        "model": "BAAI/bge-large-zh-v1.5",
                        "input": text,
                        "encoding_format": "float"
                    }
                    
                    headers = {
                        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    
                    try:
                        response = requests.post(url, json=payload, headers=headers)
                        
                        if response.status_code == 200:
                            result = response.json()
                            if "data" in result and len(result["data"]) > 0 and "embedding" in result["data"][0]:
                                embedding = np.array(result["data"][0]["embedding"])
                                batch_embeddings.append(embedding)
                            else:
                                print(f"API响应格式错误: {result}")
                                # 使用零向量作为回退
                                if len(batch_embeddings) > 0:
                                    # 使用与之前向量相同维度的零向量
                                    batch_embeddings.append(np.zeros_like(batch_embeddings[0]))
                                else:
                                    # 假设维度为1024（bge-large-zh-v1.5的维度）
                                    batch_embeddings.append(np.zeros(1024))
                        else:
                            print(f"API请求失败: {response.status_code}, {response.text}")
                            # 使用零向量作为回退
                            if len(batch_embeddings) > 0:
                                batch_embeddings.append(np.zeros_like(batch_embeddings[0]))
                            else:
                                batch_embeddings.append(np.zeros(1024))
                    except Exception as e:
                        print(f"API请求异常: {str(e)}")
                        # 使用零向量作为回退
                        if len(batch_embeddings) > 0:
                            batch_embeddings.append(np.zeros_like(batch_embeddings[0]))
                        else:
                            batch_embeddings.append(np.zeros(1024))
                
                # 将批次结果添加到总结果中
                if batch_embeddings:
                    all_embeddings.append(np.array(batch_embeddings))
                
                # 手动触发垃圾回收
                import gc
                gc.collect()
            
            # 合并所有批次的结果
            if len(all_embeddings) == 1:
                return all_embeddings[0]
            else:
                return np.vstack(all_embeddings)
    except Exception as e:
        print(f"获取嵌入向量时出错: {str(e)}")
        return None

def send_message(message, history_id, deep_thinking=False, web_search=False, doc_search=False, doc_category='clients'):
    try:
        print(f"\n=== 消息处理开始 ===")
        print(f"用户消息: {message}")
        print(f"历史记录ID: {history_id}")
        print(f"深度思考模式: {deep_thinking}")
        print(f"网络搜索模式: {web_search}")
        print(f"文档检索模式: {doc_search}")
        print(f"文档类别: {doc_category}")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT content FROM chat_history WHERE id = ?', (history_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return jsonify({'error': '会话不存在'}), 404
            
        content = row['content']
        chat_history = json.loads(content) if content else []
        
        # 添加用户消息到历史记录
        user_message = {
            'role': 'user',
            'content': message
        }
        chat_history.append(user_message)
        
        # 设置系统消息
        system_message = {
            'role': 'system',
            'content': '你是申斯小斯，一个专业、友好的AI助手。'
        }
        
        if deep_thinking:
            system_message['content'] += '\n请先分析用户的问题，思考解决方案，然后提供详细的回答。使用<think>标签包裹你的思考过程。'
            
        if web_search:
            system_message['content'] += '\n请使用网络搜索结果来帮助回答问题。'
            
        if doc_search:
            system_message['content'] += f'\n请根据用户在{doc_category}分类下的文档内容来回答问题。'
        
        # 构建消息列表
        messages = [system_message] + chat_history
        
        # 限制上下文长度
        messages = limit_context(messages)
        
        # 如果启用了网络搜索
        search_results = []
        if web_search:
            search_results = brave_web_search(message)
            if search_results:
                # 将搜索结果添加到系统消息中
                formatted_results = format_search_results(search_results)
                search_message = {
                    'role': 'system',
                    'content': f'以下是相关的搜索结果，请据此回答用户问题：\n\n{formatted_results}'
                }
                messages.append(search_message)
                
        # 如果启用了文档检索
        doc_results = []
        if doc_search:
            # 检查该分类下是否有文档
            category_path = os.path.join(VECTOR_DB_DIR, doc_category)
            has_documents = False
            
            if os.path.exists(category_path) and os.listdir(category_path):
                has_documents = True
                # 获取文档嵌入向量
                query_embedding = get_embeddings([message])[0]
                
                # 搜索每个文档集合
                all_results = []
                for collection_name in os.listdir(category_path):
                    collection_path = os.path.join(category_path, collection_name)
                    if os.path.isdir(collection_path) and os.path.exists(os.path.join(collection_path, 'index.faiss')):
                        try:
                            # 加载向量数据库
                            vector_db = VectorDB.load(collection_name, doc_category)
                            # 搜索相似文档
                            results = vector_db.search(query_embedding, k=3)
                            all_results.extend(results)
                        except Exception as e:
                            print(f"搜索文档时出错: {str(e)}")
                
                # 按相似度排序并获取前5个结果
                all_results.sort(key=lambda x: x['score'], reverse=False)  # 分数越低越相似
                doc_results = all_results[:5]
                
                if doc_results:
                    # 将文档内容添加到系统消息中
                    doc_content = ""
                    for i, result in enumerate(doc_results):
                        doc_content += f"[{i+1}] {result['text']}\n\n"
                    
                    doc_message = {
                        'role': 'system',
                        'content': f'以下是来自"{doc_category}"分类的相关文档内容，请据此回答用户问题：\n\n{doc_content}'
                    }
                    messages.append(doc_message)
            
            if not has_documents:
                # 如果没有文档，告知用户
                assistant_message = {
                    'role': 'assistant',
                    'content': f'抱歉，您在"{doc_category}"分类下没有任何文档。请先上传相关文档，然后再使用文档检索功能。'
                }
                chat_history.append(assistant_message)
                
                # 更新聊天历史
                cursor.execute('UPDATE chat_history SET content = ? WHERE id = ?',
                              (json.dumps(chat_history), history_id))
                conn.commit()
                conn.close()
                
                return jsonify({'response': assistant_message['content']})
        
        try:
            # 生成模型回复
            chat_completion = client.chat.completions.create(
                model="llama3-70b-8192",  # 使用Groq的llama3模型
                messages=[
                    {"role": m["role"], "content": m["content"]} 
                    for m in messages
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            # 提取回复
            assistant_reply = chat_completion.choices[0].message.content
            
            # 构建助手的回复
            assistant_message = {
                'role': 'assistant',
                'content': assistant_reply
            }
            
            # 添加到历史记录
            chat_history.append(assistant_message)
            
            # 如果是第一条消息，更新聊天标题
            if len(chat_history) == 2:  # 用户消息和助手回复
                title = generate_title(message)
                cursor.execute('UPDATE chat_history SET title = ? WHERE id = ?', 
                              (title, history_id))
            
            # 更新聊天历史
            cursor.execute('UPDATE chat_history SET content = ? WHERE id = ?',
                          (json.dumps(chat_history), history_id))
            conn.commit()
            conn.close()
            
            print(f"=== 消息处理完成 ===\n")
            return jsonify({'response': assistant_reply})
            
        except Exception as e:
            print(f"调用AI模型时出错: {str(e)}")
            conn.close()
            return jsonify({'error': f'生成回复时出错: {str(e)}'}), 500
            
    except Exception as e:
        print(f"处理消息时出错: {str(e)}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# 添加用户文档上传端点
@app.route('/api/upload_document', methods=['POST'])
@login_required
def upload_document():
    import gc  # 导入垃圾回收模块
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
            
        if not file.filename.endswith('.txt'):
            return jsonify({'error': '目前只支持txt格式文件'}), 400
            
        # 检查API密钥是否存在
        if not SILICONFLOW_API_KEY:
            return jsonify({'error': '无法加载任何嵌入模型，请检查环境变量中是否设置了SILICONFLOW_API_KEY。'}), 400
        
        # 获取表单参数
        category = request.form.get('category', 'clients')
        chunk_size = int(request.form.get('chunk_size', 400))
        overlap = int(request.form.get('overlap', 50))
        batch_size = int(request.form.get('batch_size', 16))  # 批处理大小
        
        # 验证参数
        if category not in DOC_CATEGORIES:
            return jsonify({'error': '无效的文档类别'}), 400
            
        if chunk_size > 500:
            chunk_size = 400
        
        if overlap > 100:
            overlap = 50
            
        if batch_size > 32:
            batch_size = 16
        
        # 创建文档目录
        docs_dir = os.path.join(DOCS_DIR, category)
        os.makedirs(docs_dir, exist_ok=True)
        
        # 生成唯一文件名
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        save_filename = f"{unique_id}_{filename}"
        file_path = os.path.join(docs_dir, save_filename)
        
        # 保存文件
        file.save(file_path)
        
        # 创建向量数据库
        collection_name = f"doc_{unique_id}"
        vector_db = VectorDB(collection_name, category)
        
        # 流式处理文件：按行读取并收集文本块
        chunks = []
        metadata = []
        chunk_id = 0
        current_chunk = ""
        
        # 按行读取文件，避免一次性加载整个文件到内存
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                current_chunk += line
                
                # 当达到目标块大小时处理
                if len(current_chunk) >= chunk_size:
                    # 找最近的句子结束符
                    end_pos = len(current_chunk)
                    for punct in ['.', '。', '!', '！', '?', '？', '\n']:
                        pos = current_chunk.rfind(punct, max(0, end_pos - chunk_size))
                        if pos != -1 and pos < end_pos:
                            end_pos = pos + 1
                    
                    # 添加一个完整的文本块
                    chunks.append(current_chunk[:end_pos])
                    metadata.append({
                        'user_id': current_user.id,
                        'filename': filename,
                        'chunk_id': chunk_id,
                        'total_chunks': -1  # 暂时不知道总数，后面更新
                    })
                    chunk_id += 1
                    
                    # 保留带有重叠的部分
                    current_chunk = current_chunk[max(0, end_pos - overlap):]
                    
                    # 如果收集了足够的块，就处理一批
                    if len(chunks) >= batch_size:
                        try:
                            # 批量处理文本块
                            vector_db.add_documents(chunks, metadata, batch_size=batch_size)
                        except Exception as e:
                            print(f"处理文本块时出错: {str(e)}")
                            return jsonify({'error': f'无法加载任何嵌入模型，请检查网络连接或手动下载模型。如果文档过大，请尝试减小分段大小。详细错误: {str(e)}'}), 500
                        print(f"已处理 {chunk_id} 个文本块")
                        
                        # 清空批次处理列表
                        chunks = []
                        metadata = []
                        
                        # 强制垃圾回收
                        gc.collect()
        
        # 处理最后一个文本块
        if current_chunk:
            chunks.append(current_chunk)
            metadata.append({
                'user_id': current_user.id,
                'filename': filename,
                'chunk_id': chunk_id,
                'total_chunks': -1
            })
            chunk_id += 1
        
        # 处理剩余的块
        if chunks:
            # 更新总块数
            for meta in metadata:
                meta['total_chunks'] = chunk_id
                
            try:
                # 处理最后一批
                vector_db.add_documents(chunks, metadata, batch_size=batch_size)
            except Exception as e:
                print(f"处理最后一批文档时出错: {str(e)}")
                return jsonify({'error': f'无法加载任何嵌入模型，请检查网络连接或手动下载模型。如果文档过大，请尝试减小分段大小。详细错误: {str(e)}'}), 500
            
            # 清空临时存储
            chunks = []
            metadata = []
            
        # 再次强制垃圾回收
        gc.collect()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'category': category,
            'chunks': chunk_id,
            'collection_name': collection_name
        })
    
    except Exception as e:
        print(f"上传文档时出错: {str(e)}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# 获取用户所有文档
@app.route('/api/user_documents', methods=['GET'])
@login_required
def get_user_documents():
    try:
        documents = []
        
        # 遍历所有分类
        for category in DOC_CATEGORIES:
            category_path = os.path.join(VECTOR_DB_DIR, category)
            
            if not os.path.exists(category_path):
                continue
                
            # 遍历分类下的所有集合
            for collection_name in os.listdir(category_path):
                collection_path = os.path.join(category_path, collection_name)
                info_path = os.path.join(collection_path, 'info.json')
                
                # 只检查info文件，避免加载大型元数据文件
                if os.path.isdir(collection_path) and os.path.exists(info_path):
                    try:
                        # 读取信息文件（小文件）
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                        
                        # 检查info中是否有用户信息
                        user_id = None
                        filename = '未知文件'
                        
                        # 从info直接检查用户ID (优先)
                        if 'user_id' in info:
                            user_id = info.get('user_id')
                            filename = info.get('filename', '未知文件')
                        # 从metadata_sample检查
                        elif 'metadata_sample' in info and 'user_id' in info.get('metadata_sample', {}):
                            user_id = info['metadata_sample'].get('user_id')
                            filename = info['metadata_sample'].get('filename', '未知文件')
                        else:
                            # 回退到检查metadata文件第一条记录 (只在必要时)
                            metadata_path = os.path.join(collection_path, 'metadata.pkl')
                            if os.path.exists(metadata_path):
                                try:
                                    # 使用pickle的load_truncated功能尝试只读取文件开头部分
                                    with open(metadata_path, 'rb') as f:
                                        # 只读取少量数据进行检查
                                        metadata_start = pickle.load(f)
                                        if metadata_start and len(metadata_start) > 0:
                                            user_id = metadata_start[0].get('user_id')
                                            filename = metadata_start[0].get('filename', '未知文件')
                                except Exception as e:
                                    print(f"读取metadata时出错: {str(e)}")
                        
                        # 检查是否属于当前用户
                        if user_id is not None and user_id == current_user.id:
                            documents.append({
                                'collection_name': collection_name,
                                'filename': filename,
                                'category': category,
                                'count': info.get('count', 0),
                                'created_at': info.get('created_at', '')
                            })
                    except Exception as e:
                        print(f"读取文档信息时出错: {str(e)}")
        
        # 按创建时间排序
        documents.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify(documents)
        
    except Exception as e:
        print(f"获取用户文档时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 消息发送路由
@app.route('/send_message', methods=['POST'])
@login_required
def handle_message():
    data = request.json
    message = data.get('message', '')
    history_id = data.get('history_id')
    deep_thinking = data.get('deep_thinking', False)
    web_search = data.get('web_search', False)
    doc_search = data.get('doc_search', False)
    doc_category = data.get('doc_category', 'clients')
    
    if not message or not history_id:
        return jsonify({'error': '缺少必要参数'}), 400
        
    return send_message(message, history_id, deep_thinking, web_search, doc_search, doc_category)

if __name__ == '__main__':
    app.run(debug=True)
