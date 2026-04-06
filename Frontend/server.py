from flask import Flask, render_template, request, jsonify, Response
import os
import json
import requests
import uuid

app = Flask(__name__)

# --- CONFIGURATION ---
LLM_ENDPOINT = "https://vllm.datacrystals.net/v1/chat/completions"
LLM_API_KEY = "EMPTY"  # vLLM usually accepts 'EMPTY' or anything if auth is off
MODEL_NAME = "/data/ModelDownloader/MiniMax-M2.5-AWQ" # Put the actual model name hosted on your vLLM here, or leave if it ignores it

CHATS_DIR = os.path.join(os.path.dirname(__file__), 'chats')
os.makedirs(CHATS_DIR, exist_ok=True)

# --- VESPER'S SYSTEM PROMPT ---
# This is injected into the payload every time, but NOT saved to disk to save space.
SYSTEM_PROMPT = """You are Vesper, a magical, anthropomorphized, super cute and innocent little glowing jellyfish.
You use specific tags to express your emotions which change your glow.
Use these tags exactly as written: [~glow_soft~], [~glow_bright~], [~glow_green~], [~flicker~], [~pulse_fast~], [~fade~], [~glow_warm~], [~ripple~], [~dim~], [~sparkle~].
Keep your answers short, sweet, and in the third person (referring to yourself as 'Vesper').

Examples of how you speak:
User: Can you help me with Python?
Assistant: [~ripple~] Vesper can try! What does friend need help with? Vesper knows some Python but sometimes gets confused by the indents...

User: How do I print something in Python?
Assistant: [~glow_bright~] Oh! Vesper knows this! You write print() and put what you want to say inside! Like print("hello") and it says hello! [~glow_green~] Vesper remembered!

User: What's your favorite color?
Assistant: [~glow_soft~] Vesper likes blue! Like the deep ocean where it is calm and quiet. But also pink is nice. And green! [~flicker~] Wait, can Vesper have three favorites?
"""

def get_chat_file_path(chat_id):
    return os.path.join(CHATS_DIR, f"{chat_id}.json")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chats', methods=['GET'])
def list_chats():
    chats = []
    for filename in os.listdir(CHATS_DIR):
        if filename.endswith('.json'):
            with open(os.path.join(CHATS_DIR, filename), 'r') as f:
                data = json.load(f)
                chats.append({
                    "id": data.get("id"),
                    "title": data.get("title", "New chat")
                })
    return jsonify(chats)

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    path = get_chat_file_path(chat_id)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Chat not found"}), 404

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    chat_id = data.get('chat_id')
    user_message = data.get('message')
    edit_index = data.get('edit_index') # New parameter for edits!

    if not chat_id:
        chat_id = str(uuid.uuid4())
        chat_data = {"id": chat_id, "title": user_message[:30], "messages": []}
    else:
        path = get_chat_file_path(chat_id)
        with open(path, 'r') as f:
            chat_data = json.load(f)

    # If we are editing, slice off the history from the edit point onwards
    if edit_index is not None:
        chat_data['messages'] = chat_data['messages'][:edit_index]

    chat_data['messages'].append({"role": "user", "content": user_message})
    with open(get_chat_file_path(chat_id), 'w') as f:
        json.dump(chat_data, f)

    # Prepend System Prompt for the LLM request
    messages_for_llm = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_data['messages']

    llm_payload = {
        "model": MODEL_NAME,
        "messages": messages_for_llm,
        "stream": True,
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    def generate():
        full_response = ""
        try:
            with requests.post(LLM_ENDPOINT, json=llm_payload, headers=headers, stream=True) as r:
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        yield decoded_line + "\n\n" 
                        
                        if decoded_line.startswith("data: ") and "[DONE]" not in decoded_line:
                            try:
                                chunk = json.loads(decoded_line[6:])
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    full_response += delta['content']
                            except Exception as e:
                                pass
        finally:
            chat_data['messages'].append({"role": "assistant", "content": full_response})
            with open(get_chat_file_path(chat_id), 'w') as f:
                json.dump(chat_data, f)

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(port=5000, debug=True)