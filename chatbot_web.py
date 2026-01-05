import http.server
import socketserver
import json
import re
import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os

# --- Bagian 1: Chatbot Engine ---
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    slang = {
        'gak': 'tidak', 'nggak': 'tidak', 'ga': 'tidak', 'gk': 'tidak',
        'makasih': 'terima kasih', 'makasi': 'terima kasih', 'thx': 'terima kasih',
        'udah': 'sudah', 'udh': 'sudah',
        'nyampe': 'sampai', 'nyampai': 'sampai', 'sampe': 'sampai', 'nyampee': 'sampai',
        'brg': 'barang', 'pesen': 'pesan', 'pesanen': 'pesanan',
        'anjir': 'kesal', 'anjing': 'kesal',
        'min': '', 'kak': '', 'bro': '', 'mbak': '', 'mas': '',
        'gua': 'saya', 'gw': 'saya', 'lu': 'kamu', 'elo': 'kamu',
        'bgt': 'banget', 'bangettt': 'banget',
        'dah': 'sudah', 'deh': '', 'nih': '', 'dong': '', 'sih': '',
        'mah': '', 'ah': '', 'eh': '', 'loh': '','aja': '', 'ajah': '',
        'klo': 'kalau', 'klw': 'kalau','sm': 'sama', 'sma': 'sama',
        'bisa2': 'bisa saja', 'bisa aja': 'bisa saja','gitu2an': 'begituan',
        'dikit2': 'sedikit-sedikit','lama2': 'lama-lama',
        'ntar': 'nanti','btw': 'omong-omong',
        'idk': 'saya tidak tahu', 'ikr': 'saya tahu',
        'tq': 'terima kasih', 'ty': 'terima kasih',
        'yw': 'sama-sama','sama2': 'sama-sama',
        'plis': 'tolong','kok': 'kenapa',
        'nih': '', 'deh': '','yaudah': 'ya sudah',
        'yuk': 'ayo','ayo2': 'ayo saja',
        'santai aja': 'santai saja','gpp': 'tidak apa-apa', 'gapapa': 'tidak apa-apa',
        'lama banget': 'sangat lama',
        'mau retur': 'ingin retur','cara retur': 'metode retur','gimana': 'bagaimana',
        'bisa gak': 'bisa tidak','yak': 'iya','bot': 'robot','kamu': 'kamu','ya': 'iya',
        'ini': 'ini','sampe': 'sampai','udah': 'sudah','belom': 'belum',
        'halo': 'halo','helo': 'halo','heloo': 'halo','hai': 'halo','hi': 'halo','complain': 'komplain',
        'mau': 'ingin'
    }
    for s, f in slang.items():
        text = text.replace(s, f)
    tokens = [w for w in text.split() if w not in {
        'yang', 'di', 'ke', 'dari', 'dan', 'adalah', 'untuk', 'dengan',
        'itu', 'ada', 'juga', 'sudah', 'akan', 'bisa', 'boleh',
        'nya', 'nih', 'dong', 'sih', 'deh', 'mah', 'ah', 'eh', 'loh',
        'aja', 'ajah', 'banget'
    } and len(w) > 1]
    return stemmer.stem(' '.join(tokens))

# --- Load atau Train Model ---
try:
    model = joblib.load('chatbot_model.pkl')  # Load model
    vectorizer = joblib.load('vectorizer.pkl')  # Load vectorizer
    with open('intents.json', encoding='utf-8') as f:
        data = json.load(f)  # Load intents
except Exception as e:
    # Training cepat (hanya jika file tidak ada)
    print("Error loading model or data:", e)
    with open('intents.json', encoding='utf-8') as f:
        data = json.load(f)
    X, y = [], []
    for intent in data['intents']:
        for p in intent['patterns']:
            X.append(preprocess(p))
            y.append(intent['tag'])
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2), max_df=0.95)
    Xv = vectorizer.fit_transform(X)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(Xv, y)
    joblib.dump(model, 'chatbot_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

def get_bot_response(user_input):
    try:
        vec = vectorizer.transform([preprocess(user_input)])
        tag = model.predict(vec)[0]
        conf = model.predict_proba(vec).max()
        if conf < 0.2:
            return "Maaf, saya kurang paham. Bisa dijelaskan lebih detail?"
        for intent in data['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    except Exception as e:
        return "Mohon maaf, terjadi kesalahan. Error: " + str(e)
    return "Maaf, saya belum paham."

# --- Bagian 2: HTTP Handler ---
class Handler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        # Handle preflight request (CORS)
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('public/index.html', 'r') as f:
                self.wfile.write(f.read().encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/chat':
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                post_body = self.rfile.read(content_len)
                data = json.loads(post_body)
                response = get_bot_response(data.get('message', ''))
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'response': response}).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'{"response":"Error"}')
        else:
            self.send_response(404)
            self.end_headers()

# --- Jalankan Server di Port 8000 ---
if __name__ == '__main__':
    print("🚀 Server berjalan di http://localhost:8000")
    print("Tekan Ctrl+C untuk berhenti.")
    with socketserver.TCPServer(("", 8000), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server dihentikan.")
