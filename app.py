
from flask import Flask, render_template, jsonify, request, session, redirect, url_for, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import cv2
import base64
import threading
import time
import sys
import os
import sqlite3
import bcrypt
import secrets
from functools import wraps
from datetime import timedelta

app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'),
            static_url_path='/static')

# ── Sessoes seguras ────────────────────────────────────────────────────────────
app.secret_key = os.environ.get('SIGEA_SECRET', secrets.token_hex(32))
app.config.update(
    SESSION_COOKIE_HTTPONLY  = True,   # JS nao acede ao cookie
    SESSION_COOKIE_SAMESITE  = 'Lax',  # protecao CSRF basica
    SESSION_COOKIE_SECURE    = False,  # mudar para True com HTTPS
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8),
)

CORS(app, supports_credentials=True)

# ── Rate limiting ──────────────────────────────────────────────────────────────
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://"
)

from recohecimeto_gestos import GestureRecognizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'sigea.db')

recognizer        = GestureRecognizer()
cap               = None
is_capturing      = False
current_frame_b64 = None
current_gesture   = None
lock              = threading.Lock()
cap_lock          = threading.Lock()


# ══ BASE DE DADOS ══════════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS utilizadores (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                email           TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                senha_hash      TEXT    NOT NULL,
                role            TEXT    NOT NULL DEFAULT 'aluno'
                                CHECK(role IN ('admin','aluno')),
                ativo           INTEGER NOT NULL DEFAULT 1,
                tentativas_fail INTEGER NOT NULL DEFAULT 0,
                bloqueado_ate   TEXT,
                criado_em       TEXT    DEFAULT (datetime('now')),
                ultimo_login    TEXT
            );

            CREATE TABLE IF NOT EXISTS login_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER REFERENCES utilizadores(id) ON DELETE SET NULL,
                email      TEXT,
                ip         TEXT,
                sucesso    INTEGER,
                criado_em  TEXT DEFAULT (datetime('now'))
            );
        """)

    # Criar admin por defeito se nao existir
    with get_db() as conn:
        admin = conn.execute("SELECT id FROM utilizadores WHERE role='admin'").fetchone()
        if not admin:
            senha_hash = bcrypt.hashpw('mariel22'.encode(), bcrypt.gensalt()).decode()
            conn.execute(
                "INSERT INTO utilizadores (email, senha_hash, role) VALUES (?,?,'admin')",
                ('admin@sigea.ao', senha_hash)
            )
            print("[DB] Administrador criado — email: admin@sigea.ao | senha: mariel22")

    print(f"[DB] Base de dados pronta: {DB_PATH}")


# ── Senhas com bcrypt ──────────────────────────────────────────────────────────

def hash_senha(senha: str) -> str:
    return bcrypt.hashpw(senha.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verificar_senha(senha: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(senha.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False


# ── Bloqueio por tentativas falhadas ──────────────────────────────────────────
MAX_TENTATIVAS = 5
BLOQUEIO_MIN   = 15  # minutos


def registar_tentativa(user_id, email, ip, sucesso):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO login_log (user_id, email, ip, sucesso) VALUES (?,?,?,?)",
            (user_id, email, ip, 1 if sucesso else 0)
        )
        if user_id:
            if sucesso:
                conn.execute(
                    "UPDATE utilizadores SET tentativas_fail=0, bloqueado_ate=NULL, ultimo_login=datetime('now') WHERE id=?",
                    (user_id,)
                )
            else:
                conn.execute(
                    """UPDATE utilizadores
                       SET tentativas_fail = tentativas_fail + 1,
                           bloqueado_ate = CASE
                               WHEN tentativas_fail + 1 >= ? THEN datetime('now', '+' || ? || ' minutes')
                               ELSE bloqueado_ate
                           END
                       WHERE id=?""",
                    (MAX_TENTATIVAS, BLOQUEIO_MIN, user_id)
                )


def conta_bloqueada(user) -> bool:
    if not user['bloqueado_ate']:
        return False
    from datetime import datetime
    bloqueado = datetime.fromisoformat(user['bloqueado_ate'])
    return datetime.utcnow() < bloqueado


# ══ CAMERA ════════════════════════════════════════════════════════════════════

def camera_thread():
    global cap, is_capturing, current_frame_b64, current_gesture
    consecutive_failures = 0
    while True:
        with lock:
            capturing = is_capturing
        if not capturing:
            time.sleep(0.05); consecutive_failures = 0; continue
        with cap_lock:
            local_cap = cap
        if local_cap is None:
            time.sleep(0.05); continue
        ret, frame = local_cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= 30:
                with lock:
                    is_capturing = False; current_frame_b64 = None; current_gesture = None
                consecutive_failures = 0
            time.sleep(0.033); continue
        consecutive_failures = 0
        frame = cv2.resize(frame, (640, 480))
        frame, gesture_result, _ = recognizer.process_frame(frame)
        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok: continue
        with lock:
            current_frame_b64 = base64.b64encode(buffer).decode('utf-8')
            current_gesture   = gesture_result


camera_worker = threading.Thread(target=camera_thread, daemon=True, name="CameraThread")
camera_worker.start()


# ══ DECORADORES ═══════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth_page'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth_page'))
        if session.get('user_role') != 'admin':
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated


def api_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'message': 'Nao autenticado.'}), 401
        return f(*args, **kwargs)
    return decorated


def api_admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'message': 'Nao autenticado.'}), 401
        if session.get('user_role') != 'admin':
            return jsonify({'message': 'Acesso restrito.'}), 403
        return f(*args, **kwargs)
    return decorated


# ══ PAGINAS ═══════════════════════════════════════════════════════════════════

@app.route('/auth')
def auth_page():
    if 'user_id' in session:
        return redirect(url_for('admin_page') if session.get('user_role') == 'admin' else url_for('index'))
    return render_template('auth.html')


@app.route('/admin')
@admin_required
def admin_page():
    return render_template('admin.html')


@app.route('/')
@login_required
def index():
    return render_template('index.html')


@app.route('/dicionario')
@login_required
def dicionario():
    return render_template('dicionario.html')


@app.route('/treinamento')
@login_required
def treinamento():
    return render_template('treinamento.html')


@app.route('/sobre')
@login_required
def sobre():
    return render_template('sobre.html')


# ══ AUTENTICACAO ══════════════════════════════════════════════════════════════

@app.route('/api/auth/registo', methods=['POST'])
@limiter.limit("5 per hour")   # max 5 registos por hora por IP
def api_registo():
    data  = request.get_json(force=True)
    email = (data.get('email') or '').strip().lower()
    senha = data.get('senha') or ''

    if not email or '@' not in email or '.' not in email:
        return jsonify({'message': 'Email invalido.'}), 400
    if len(senha) < 6:
        return jsonify({'message': 'A senha deve ter pelo menos 6 caracteres.'}), 400

    try:
        with get_db() as conn:
            if conn.execute("SELECT id FROM utilizadores WHERE email=?", (email,)).fetchone():
                return jsonify({'message': 'Email ja registado.'}), 409
            conn.execute(
                "INSERT INTO utilizadores (email, senha_hash, role) VALUES (?,?,'aluno')",
                (email, hash_senha(senha))
            )
        return jsonify({'message': 'Conta criada com sucesso!'}), 201
    except sqlite3.Error as e:
        return jsonify({'message': 'Erro interno no servidor.'}), 500


@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10 per minute")  # max 10 tentativas por minuto por IP
def api_login():
    data  = request.get_json(force=True)
    email = (data.get('email') or '').strip().lower()
    senha = data.get('senha') or ''
    role_pretendido = data.get('role', 'aluno')
    ip = request.remote_addr

    if not email or not senha:
        return jsonify({'message': 'Email e senha sao obrigatorios.'}), 400

    try:
        with get_db() as conn:
            user = conn.execute(
                "SELECT id, email, senha_hash, role, ativo, tentativas_fail, bloqueado_ate FROM utilizadores WHERE email=?",
                (email,)
            ).fetchone()

        # Utilizador nao existe — resposta generica para nao revelar info
        if not user:
            return jsonify({'message': 'Credenciais incorretas.'}), 401

        # Conta bloqueada por tentativas excessivas
        if conta_bloqueada(user):
            return jsonify({'message': f'Conta temporariamente bloqueada. Tenta em {BLOQUEIO_MIN} minutos.'}), 429

        # Verificar senha com bcrypt
        if not verificar_senha(senha, user['senha_hash']):
            registar_tentativa(user['id'], email, ip, False)
            tentativas_restantes = MAX_TENTATIVAS - (user['tentativas_fail'] + 1)
            if tentativas_restantes <= 0:
                return jsonify({'message': f'Conta bloqueada por {BLOQUEIO_MIN} min apos muitas tentativas falhadas.'}), 429
            return jsonify({'message': f'Credenciais incorretas. {max(tentativas_restantes,0)} tentativa(s) restante(s).'}), 401

        if not user['ativo']:
            return jsonify({'message': 'Conta desativada. Contacta o professor.'}), 403

        # Verificar role
        if role_pretendido == 'admin' and user['role'] != 'admin':
            return jsonify({'message': 'Nao tens permissoes de professor.'}), 403
        if role_pretendido == 'aluno' and user['role'] == 'admin':
            return jsonify({'message': 'Usa o perfil Professor para entrar.'}), 403

        # Login bem sucedido
        registar_tentativa(user['id'], email, ip, True)

        session.clear()
        session.permanent = True
        session['user_id']    = user['id']
        session['user_email'] = user['email']
        session['user_role']  = user['role']

        return jsonify({
            'message':  'Acesso concedido.',
            'email':    user['email'],
            'role':     user['role'],
            'redirect': '/admin' if user['role'] == 'admin' else '/'
        }), 200

    except sqlite3.Error as e:
        print(f"[DB] Erro login: {e}", file=sys.stderr)
        return jsonify({'message': 'Erro interno no servidor.'}), 500


@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'message': 'Sessao terminada.'}), 200


@app.route('/api/auth/me', methods=['GET'])
def api_me():
    if 'user_id' not in session:
        return jsonify({'autenticado': False}), 401
    return jsonify({
        'autenticado': True,
        'email':       session.get('user_email'),
        'role':        session.get('user_role')
    }), 200


# ══ API ADMIN ═════════════════════════════════════════════════════════════════

@app.route('/api/admin/alunos', methods=['GET'])
@api_admin_required
def api_admin_alunos():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, email, role, ativo, tentativas_fail, criado_em, ultimo_login FROM utilizadores WHERE role='aluno' ORDER BY criado_em DESC"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/admin/alunos/<int:uid>', methods=['DELETE'])
@api_admin_required
def api_admin_apagar_aluno(uid):
    with get_db() as conn:
        user = conn.execute("SELECT role FROM utilizadores WHERE id=?", (uid,)).fetchone()
        if not user:
            return jsonify({'message': 'Utilizador nao encontrado.'}), 404
        if user['role'] == 'admin':
            return jsonify({'message': 'Nao podes eliminar um administrador.'}), 403
        conn.execute("DELETE FROM utilizadores WHERE id=?", (uid,))
    return jsonify({'message': 'Conta eliminada.'}), 200


@app.route('/api/admin/alunos/<int:uid>/toggle', methods=['POST'])
@api_admin_required
def api_admin_toggle_aluno(uid):
    with get_db() as conn:
        user = conn.execute("SELECT ativo, role FROM utilizadores WHERE id=?", (uid,)).fetchone()
        if not user:
            return jsonify({'message': 'Utilizador nao encontrado.'}), 404
        if user['role'] == 'admin':
            return jsonify({'message': 'Nao podes desativar um administrador.'}), 403
        novo = 0 if user['ativo'] else 1
        conn.execute("UPDATE utilizadores SET ativo=?, tentativas_fail=0, bloqueado_ate=NULL WHERE id=?", (novo, uid))
    return jsonify({'message': 'Estado atualizado.', 'ativo': novo}), 200


@app.route('/api/admin/alunos/<int:uid>/desbloquear', methods=['POST'])
@api_admin_required
def api_admin_desbloquear(uid):
    with get_db() as conn:
        conn.execute(
            "UPDATE utilizadores SET tentativas_fail=0, bloqueado_ate=NULL WHERE id=?", (uid,)
        )
    return jsonify({'message': 'Conta desbloqueada.'}), 200


@app.route('/api/admin/videos', methods=['GET'])
@api_admin_required
def api_admin_videos():
    videos_dir = os.path.join(BASE_DIR, 'static', 'videos')
    if not os.path.exists(videos_dir):
        return jsonify([])
    return jsonify([f for f in os.listdir(videos_dir) if f.endswith('.mp4')])


@app.route('/api/admin/logs', methods=['GET'])
@api_admin_required
def api_admin_logs():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT email, ip, sucesso, criado_em FROM login_log ORDER BY criado_em DESC LIMIT 100"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


# ══ API CAMERA ════════════════════════════════════════════════════════════════

@app.route('/api/start-capture', methods=['POST'])
@api_login_required
def start_capture():
    global is_capturing, cap
    with cap_lock:
        if cap is None:
            new_cap = cv2.VideoCapture(0)
            if not new_cap.isOpened():
                return jsonify({'status': 'error', 'message': 'Camera nao encontrada.'}), 400
            new_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            new_cap.set(cv2.CAP_PROP_FPS,          15)
            cap = new_cap
    with lock:
        is_capturing = True
    return jsonify({'status': 'success'})


@app.route('/api/stop-capture', methods=['POST'])
@api_login_required
def stop_capture():
    global is_capturing, cap, current_frame_b64, current_gesture
    with lock:
        is_capturing = False; current_frame_b64 = None; current_gesture = None
    time.sleep(0.15)
    with cap_lock:
        if cap is not None:
            cap.release(); cap = None
    return jsonify({'status': 'stopped'})


@app.route('/api/get-frame', methods=['GET'])
@api_login_required
def get_frame():
    with lock:
        frame_b64 = current_frame_b64
        gesture   = current_gesture
    if not frame_b64:
        return jsonify({'error': 'Sem frame disponivel.'}), 400
    response = {'frame': frame_b64, 'gesture': None}
    if gesture:
        response['gesture'] = {
            'gesture':    gesture.get('gesture', 'Desconhecido'),
            'confidence': float(gesture.get('confidence', 0.0)),
            'detected':   gesture.get('gesture') != 'Desconhecido'
        }
    return jsonify(response)


@app.route('/api/status')
def get_status():
    with lock:
        capturing = is_capturing; has_frame = current_frame_b64 is not None; gesture = current_gesture
    with cap_lock:
        cam_open = cap is not None and cap.isOpened() if cap else False
    return jsonify({'capturing': capturing, 'camera_open': cam_open, 'has_frame': has_frame, 'last_gesture': gesture, 'thread_alive': camera_worker.is_alive()})


@app.route('/static/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'static', 'videos'), filename)


# ══ API GESTOS ════════════════════════════════════════════════════════════════

@app.route('/api/gestos')
def get_gestos():
    return jsonify([
        {'id':1,'nome':'Punho Fechado',    'emoji':'✊','descricao':'Todos os dedos fechados',  'uso':'Forca',    'dificuldade':'Facil'},
        {'id':2,'nome':'Mao Aberta',       'emoji':'✋','descricao':'Todos os dedos abertos',   'uso':'Saudacao', 'dificuldade':'Facil'},
        {'id':3,'nome':'Apontar',          'emoji':'☝️','descricao':'Indicador levantado',      'uso':'Indicar',  'dificuldade':'Facil'},
        {'id':4,'nome':'Paz (V)',          'emoji':'✌️','descricao':'Indicador e medio abertos','uso':'Paz',      'dificuldade':'Medio'},
        {'id':5,'nome':'Polegar para Cima','emoji':'👍','descricao':'Polegar levantado',        'uso':'Aprovacao','dificuldade':'Medio'},
        {'id':6,'nome':'Mao Levantada',    'emoji':'🤚','descricao':'Mao aberta levantada',     'uso':'Saudacao', 'dificuldade':'Facil'},
    ])


# ══ ARRANQUE ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    init_db()
    print("=" * 52)
    print("  SIGEA — Lingua Gestual Angolana")
    print("  Acesso : http://localhost:5000/auth")
    print("  Admin  : http://localhost:5000/auth (Professor)")
    print("  Status : http://localhost:5000/api/status")
    print("=" * 52)
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
