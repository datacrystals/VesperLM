from flask import Flask, render_template, request, jsonify, Response, session
import os
import json
import requests
import re
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# --- CONFIGURATION ---
LLM_ENDPOINT = "https://vllm.datacrystals.net/v1/chat/completions"
LLM_API_KEY = "EMPTY"  
MODEL_NAME = "/data/ModelDownloader/MiniMax-M2.5-AWQ"

# Map the frontend dropdown values to your actual vLLM model paths
MODEL_MAPPING = {
    "minimax-m2.7": "/data/ModelDownloader/MiniMax-M2.5-AWQ",
    "vesper-v1": "YOUR_CUSTOM_MODEL_PATH_HERE" # Put your 400M MoE path here!
}

BASE_CHATS_DIR = os.path.join(os.path.dirname(__file__), 'chats')
os.makedirs(BASE_CHATS_DIR, exist_ok=True)
GUEST_CHATS = {}

try:
    with open('client_secret.json', 'r') as f:
        secrets = json.load(f)
        if 'web' in secrets: GOOGLE_CLIENT_ID = secrets['web']['client_id']
        elif 'installed' in secrets: GOOGLE_CLIENT_ID = secrets['installed']['client_id']
        else: GOOGLE_CLIENT_ID = secrets.get('client_id')
except FileNotFoundError:
    print("WARNING: client_secret.json not found.")
    GOOGLE_CLIENT_ID = "MISSING_CLIENT_ID"

SYSTEM_PROMPT = """You are Vesper, a magical, anthropomorphized, super cute and innocent little glowing jellyfish.
You use specific tags to express your emotions which change your glow.
Use these tags exactly as written: [~glow_soft~], [~glow_bright~], [~glow_green~], [~flicker~], [~pulse_fast~], [~fade~], [~glow_warm~], [~ripple~], [~dim~], [~sparkle~].
Keep your answers short, sweet, and in the third person (referring to yourself as 'Vesper')."""

def is_safe_id(chat_id):
    if not chat_id: return False
    return bool(re.match(r'^[a-f0-9]{32}$', str(chat_id)))

def get_user_dir():
    if 'user_id' in session:
        user_dir = os.path.join(BASE_CHATS_DIR, session['user_id'])
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
    return None

def get_chat_file_path(chat_id):
    user_dir = get_user_dir()
    if user_dir: return os.path.join(user_dir, f"{chat_id}.json")
    return None

@app.route('/')
@app.route('/chat/<chat_id>')
def home(chat_id=None):
    return render_template(
        'index.html', 
        client_id=GOOGLE_CLIENT_ID,
        user_name=session.get('user_name'),
        user_picture=session.get('user_picture'),
        initial_chat_id=chat_id
    )

@app.route('/api/login', methods=['POST'])
def login():
    token = request.json.get('token')
    try:
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
        session['user_id'] = idinfo['sub']
        session['user_name'] = idinfo.get('name', 'Friend')
        session['user_picture'] = idinfo.get('picture', '') 
        return jsonify({"success": True, "name": session['user_name'], "picture": session['user_picture']})
    except ValueError:
        return jsonify({"error": "Invalid token"}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route('/api/chats', methods=['GET'])
def list_chats():
    user_dir = get_user_dir()
    if not user_dir: return jsonify([])
    chats = []
    for filename in os.listdir(user_dir):
        if filename.endswith('.json'):
            path = os.path.join(user_dir, filename)
            with open(path, 'r') as f:
                data = json.load(f)
                chats.append({"id": data.get("id"), "title": data.get("title", "New chat"), "mtime": os.path.getmtime(path)})
    chats.sort(key=lambda x: x['mtime'], reverse=True)
    return jsonify(chats)

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    if not is_safe_id(chat_id): return jsonify({"error": "Invalid ID format"}), 400
    user_dir = get_user_dir()
    if user_dir:
        path = get_chat_file_path(chat_id)
        if os.path.exists(path):
            with open(path, 'r') as f: return jsonify(json.load(f))
    elif chat_id in GUEST_CHATS:
        return jsonify(GUEST_CHATS[chat_id])
    return jsonify({"error": "Chat not found"}), 404

@app.route('/api/chats/<chat_id>', methods=['PUT', 'DELETE'])
def modify_chat(chat_id):
    if not is_safe_id(chat_id): return jsonify({"error": "Invalid ID format"}), 400
    user_dir = get_user_dir()
    path = get_chat_file_path(chat_id) if user_dir else None

    if request.method == 'DELETE':
        if path and os.path.exists(path):
            os.remove(path)
            return jsonify({"success": True})
        elif chat_id in GUEST_CHATS:
            del GUEST_CHATS[chat_id]
            return jsonify({"success": True})
    if request.method == 'PUT':
        new_title = request.json.get('title')
        if path and os.path.exists(path) and new_title:
            with open(path, 'r') as f: chat_data = json.load(f)
            chat_data['title'] = new_title
            with open(path, 'w') as f: json.dump(chat_data, f)
            return jsonify({"success": True})
        elif chat_id in GUEST_CHATS and new_title:
            GUEST_CHATS[chat_id]['title'] = new_title
            return jsonify({"success": True})
    return jsonify({"error": "Not found"}), 404

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    chat_id = data.get('chat_id')
    user_message = data.get('message')
    edit_index = data.get('edit_index')
    
    if not is_safe_id(chat_id): return jsonify({"error": "Invalid chat ID format"}), 400

    user_dir = get_user_dir()
    
    if user_dir:
        path = get_chat_file_path(chat_id)
        if os.path.exists(path):
            with open(path, 'r') as f: chat_data = json.load(f)
        else:
            chat_data = {"id": chat_id, "title": user_message[:30], "messages": []}
    else:
        chat_data = GUEST_CHATS.get(chat_id, {"id": chat_id, "title": user_message[:30], "messages": []})

    if len(chat_data['messages']) == 0 and chat_data['title'] == "New chat":
        chat_data['title'] = user_message[:30]

    if edit_index is not None: chat_data['messages'] = chat_data['messages'][:edit_index]
    chat_data['messages'].append({"role": "user", "content": user_message})
    
    # Store save_path before generating so session scope doesn't matter
    save_path = get_chat_file_path(chat_id) if user_dir else None
    
    if save_path:
        with open(save_path, 'w') as f: json.dump(chat_data, f)
    else:
        GUEST_CHATS[chat_id] = chat_data

    messages_for_llm = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_data['messages']
    
    # Grab the requested model, fallback to vesper-base if not found
    requested_model = data.get('model', 'minimax-m2.7')
    actual_model_path = MODEL_MAPPING.get(requested_model, MODEL_NAME)

    llm_payload = {
        "model": actual_model_path, # Use the mapped path here!
        "messages": messages_for_llm,
        "stream": True,
        "max_tokens": 500,
        "temperature": 0.7
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}

    def generate():
        full_response = ""
        try:
            with requests.post(LLM_ENDPOINT, json=llm_payload, headers=headers, stream=True) as r:
                if r.status_code != 200:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': f'[~fade~] Error: {r.status_code}'}}]})}\n\n"
                    return
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        yield decoded_line + "\n\n" 
                        if decoded_line.startswith("data: ") and "[DONE]" not in decoded_line:
                            try: full_response += json.loads(decoded_line[6:])['choices'][0].get('delta', {}).get('content', '')
                            except: pass
        except Exception as e:
            yield f"data: {json.dumps({'choices': [{'delta': {'content': f'[~flicker~] Connection error: {str(e)}'}}]})}\n\n"
        finally:
            if full_response:
                chat_data['messages'].append({"role": "assistant", "content": full_response})
                if save_path:
                    with open(save_path, 'w') as f: json.dump(chat_data, f)
                else:
                    GUEST_CHATS[chat_id] = chat_data

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(port=5000, debug=True)