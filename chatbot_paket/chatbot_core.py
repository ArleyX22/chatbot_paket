# chatbot.py
import json
import re
import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download stopwords (jalankan sekali)
nltk.download('stopwords', quiet=True)

# === Inisialisasi Stemmer Bahasa Indonesia (Sastrawi) ===
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# === Preprocessing Teks (Optimized untuk Bahasa Indonesia Non-Formal) ===
def preprocess(text):
    # Lowercase
    text = text.lower()
    # Hapus selain huruf & spasi
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Normalisasi slang & typo umum
    slang_dict = {
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
    for slang, formal in slang_dict.items():
        text = text.replace(slang, formal)
    
    # Tokenisasi
    tokens = text.split()
    
    # Stopwords Indonesia + custom
    try:
        indo_stop = set(stopwords.words('indonesian'))
    except:
        indo_stop = set()
    custom_stop = {
        'yang', 'di', 'ke', 'dari', 'dan', 'adalah', 'untuk', 'dengan',
        'itu', 'ada', 'juga', 'sudah', 'akan', 'bisa', 'boleh',
        'nya', 'nih', 'dong', 'sih', 'deh', 'mah', 'ah', 'eh', 'loh',
        'aja', 'ajah', 'banget'
    }
    stop_words = indo_stop | custom_stop
    
    # Filter token
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    # Gabung & stem
    clean_text = ' '.join(tokens)
    stemmed = stemmer.stem(clean_text)
    return stemmed

# === Load Dataset ===
with open('intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

X, y = [], []
for intent in data['intents']:
    for pattern in intent['patterns']:
        X.append(preprocess(pattern))
        y.append(intent['tag'])

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === TF-IDF Vectorizer ===
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),  # unigram + bigram → tangkap frasa ("barang lama", "belum sampai")
    min_df=1,
    max_df=0.95,
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === Model: LogisticRegression (lebih stabil untuk data kecil) ===
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
model.fit(X_train_vec, y_train)

# === Simpan Model ===
joblib.dump(model, 'chatbot_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# === Evaluasi ===
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Akurasi: {acc:.2%}")
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Fungsi Prediksi ===
def predict_intent(text):
    processed = preprocess(text)
    vec = vectorizer.transform([processed])
    tag = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()  # ✅ sekarang bisa pakai probabilitas!
    return tag, prob

def get_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Maaf, saya belum paham. Bisa dijelaskan lebih detail?"

# === Demo Interaktif ===
print("\n" + "="*50)
print("🤖 Chatbot Layanan Paket (Versi Optimized)")
print("➡️  Bahasa non-formal & slang didukung!")
print(" Ketik 'keluar' untuk berhenti")
print("="*50)

while True:
    user_input = input("\nAnda: ").strip()
    if user_input.lower() in ['keluar', 'exit', 'quit']:
        print("👋 Terima kasih telah menggunakan layanan kami!")
        break
    if not user_input:
        continue

    try:
        tag, conf = predict_intent(user_input)
        print(f"[DEBUG] Intent: {tag}, Confidence: {conf:.2f}")
        
        if conf < 0.4:  # threshold lebih realistis
            print("🤖: Maaf, saya kurang paham. Bisa dijelaskan lebih detail?")
        else:
            print(f"🤖: {get_response(tag)}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🤖: Mohon maaf, terjadi kesalahan. Coba lagi ya.")
