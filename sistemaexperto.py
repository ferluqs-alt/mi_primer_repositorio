"""
Doctor Solución - Sistema Experto para Rinitis Alérgica
Aplicación web interactiva para niños de 6-10 años
Optimizada para Streamlit Cloud
"""

import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import os
import tempfile
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="🐻 Doctor Solución",
    page_icon="🐻",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/tu-usuario/doctor-solucion',
        'Report a bug': "https://github.com/tu-usuario/doctor-solucion/issues",
        'About': "# Doctor Solución\nSistema experto para rinitis alérgica diseñado para niños"
    }
)

# CSS personalizado para estilo infantil
st.markdown("""
<style>
    .main {
        padding: 1rem;
        background: linear-gradient(135deg, #FFE5B4 0%, #E6F3FF 100%);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    .symptom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #FF6B6B;
    }
    
    .diagnosis-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .achievement-badge {
        background: gold;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        display: inline-block;
        margin: 0.25rem;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .mission-card {
        background: #E8F5E8;
        border: 2px solid #4CAF50;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .streamlit-info {
        background: #f0f8ff;
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Animaciones CSS */
    @keyframes bounce {
        0%, 20%, 60%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        80% { transform: translateY(-5px); }
    }
    
    .animated-title {
        animation: bounce 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Funciones para manejo de base de datos en Streamlit Cloud
@st.cache_resource
def get_database_path():
    """Obtiene la ruta correcta para la base de datos según el entorno"""
    # En Streamlit Cloud, usar directorio temporal persistente
    if 'STREAMLIT_SHARING_MODE' in os.environ:
        # Streamlit Cloud environment
        db_dir = Path.home() / '.streamlit' / 'doctor_solucion'
        db_dir.mkdir(parents=True, exist_ok=True)
        return str(db_dir / 'doctor_solucion.db')
    else:
        # Local development
        return 'doctor_solucion.db'

@st.cache_resource
def init_db():
    """Inicializa la base de datos con optimizaciones para Streamlit Cloud"""
    try:
        db_path = get_database_path()
        
        # Configurar conexión con optimizaciones
        conn = sqlite3.connect(
            db_path, 
            check_same_thread=False,
            timeout=30.0,
            isolation_level=None  # Autocommit mode
        )
        
        cursor = conn.cursor()
        
        # Optimizaciones de SQLite
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.execute("PRAGMA cache_size = 1000")
        cursor.execute("PRAGMA temp_store = MEMORY")
        
        # Crear tablas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                edad INTEGER NOT NULL,
                fecha_registro DATE NOT NULL,
                session_id TEXT,
                UNIQUE(nombre, edad, session_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diagnosticos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER NOT NULL,
                sintomas TEXT NOT NULL,
                probabilidad REAL NOT NULL,
                fecha DATETIME NOT NULL,
                session_id TEXT,
                FOREIGN KEY (usuario_id) REFERENCES usuarios (id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logros (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER NOT NULL,
                tipo_logro TEXT NOT NULL,
                fecha DATETIME NOT NULL,
                session_id TEXT,
                FOREIGN KEY (usuario_id) REFERENCES usuarios (id) ON DELETE CASCADE,
                UNIQUE(usuario_id, tipo_logro)
            )
        ''')
        
        # Crear índices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_usuarios_session ON usuarios(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_diagnosticos_usuario ON diagnosticos(usuario_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logros_usuario ON logros(usuario_id)')
        
        return conn
        
    except Exception as e:
        st.error(f"Error al inicializar la base de datos: {str(e)}")
        # Fallback a base de datos en memoria
        return sqlite3.connect(':memory:', check_same_thread=False)

# Sistema de inferencia para diagnóstico
class SistemaExperto:
    def __init__(self):
        self.reglas = {
            'estornudos_frecuentes': 0.25,
            'picazon_nariz': 0.20,
            'ojos_rojos_llorosos': 0.20,
            'congestion_nasal': 0.15,
            'empeora_primavera': 0.15,
            'reaccion_mascotas': 0.15,
            'reaccion_polvo': 0.15,
            'goteo_nasal': 0.10,
            'fatiga': 0.05
        }
    
    def diagnosticar(self, sintomas):
        probabilidad = 0
        factores_presentes = []
        
        for sintoma, peso in self.reglas.items():
            if sintoma in sintomas:
                probabilidad += peso
                factores_presentes.append(sintoma)
        
        # Normalizar probabilidad (máximo 100%)
        probabilidad = min(probabilidad, 1.0)
        
        return {
            'probabilidad': probabilidad * 100,
            'factores': factores_presentes,
            'interpretacion': self._interpretar_resultado(probabilidad * 100)
        }
    
    def _interpretar_resultado(self, prob):
        if prob >= 70:
            return "¡Alto! 🔴 Es muy probable que tengas rinitis alérgica."
        elif prob >= 40:
            return "¡Cuidado! 🟡 Podrías tener rinitis alérgica."
        elif prob >= 20:
            return "¡Atención! 🟠 Algunos síntomas coinciden."
        else:
            return "¡Tranquilo! 🟢 Pocos síntomas de rinitis alérgica."

# Funciones de gestión de sesión para Streamlit Cloud
def get_session_id():
    """Genera un ID de sesión único para cada usuario"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time() * 1000)}"
    return st.session_state.session_id

def inicializar_aplicacion():
    """Inicializa todos los componentes de la aplicación"""
    if 'db' not in st.session_state:
        st.session_state.db = init_db()
    
    if 'sistema_experto' not in st.session_state:
        st.session_state.sistema_experto = SistemaExperto()
    
    if 'usuario_actual' not in st.session_state:
        st.session_state.usuario_actual = None
    
    if 'session_id' not in st.session_state:
        get_session_id()

# Funciones de base de datos con manejo de sesiones
def registrar_usuario(nombre, edad):
    """Registra un nuevo usuario en la base de datos"""
    try:
        session_id = get_session_id()
        cursor = st.session_state.db.cursor()
        
        # Verificar si el usuario ya existe en esta sesión
        cursor.execute(
            "SELECT id FROM usuarios WHERE nombre = ? AND edad = ? AND session_id = ?",
            (nombre, edad, session_id)
        )
        
        existing_user = cursor.fetchone()
        if existing_user:
            return existing_user[0]
        
        # Crear nuevo usuario
        cursor.execute(
            "INSERT INTO usuarios (nombre, edad, fecha_registro, session_id) VALUES (?, ?, ?, ?)",
            (nombre, int(edad), datetime.now().date(), session_id)
        )
        
        return cursor.lastrowid
        
    except sqlite3.Error as e:
        st.error(f"Error de base de datos: {str(e)}")
        return None

def obtener_logros_usuario(usuario_id):
    """Obtiene los logros de un usuario"""
    try:
        cursor = st.session_state.db.cursor()
        cursor.execute("SELECT tipo_logro FROM logros WHERE usuario_id = ?", (usuario_id,))
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        st.error(f"Error al obtener logros: {str(e)}")
        return []

def agregar_logro(usuario_id, tipo_logro):
    """Agrega un logro a un usuario"""
    try:
        session_id = get_session_id()
        cursor = st.session_state.db.cursor()
        
        # Verificar si el logro ya existe
        cursor.execute(
            "SELECT id FROM logros WHERE usuario_id = ? AND tipo_logro = ?", 
            (usuario_id, tipo_logro)
        )
        
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO logros (usuario_id, tipo_logro, fecha, session_id) VALUES (?, ?, ?, ?)",
                (usuario_id, tipo_logro, datetime.now(), session_id)
            )
            return True
        return False
        
    except Exception as e:
        st.error(f"Error al agregar logro: {str(e)}")
        return False

def guardar_diagnostico(usuario_id, sintomas, probabilidad):
    """Guarda un diagnóstico en la base de datos"""
    try:
        session_id = get_session_id()
        cursor = st.session_state.db.cursor()
        cursor.execute(
            "INSERT INTO diagnosticos (usuario_id, sintomas, probabilidad, fecha, session_id) VALUES (?, ?, ?, ?, ?)",
            (usuario_id, json.dumps(sintomas), probabilidad, datetime.now(), session_id)
        )
        return True
    except Exception as e:
        st.error(f"Error al guardar diagnóstico: {str(e)}")
        return False

# Inicializar aplicación
inicializar_aplicacion()

# Header con información de Streamlit Cloud
st.markdown("""
<div class="streamlit-info">
<h4>🚀 Ejecutándose en Streamlit Cloud</h4>
<p>Esta aplicación está hospedada en Streamlit Cloud con datos persistentes y optimizada para múltiples usuarios.</p>
</div>
""", unsafe_allow_html=True)

# INTERFAZ PRINCIPAL
st.markdown('<h1 class="animated-title">🐻 ¡Hola! Soy Doctor Oso, tu amigo médico</h1>', unsafe_allow_html=True)
st.markdown("### 🌟 Bienvenido a Doctor Solución - Tu aventura médica comienza aquí")

# Sidebar para navegación
with st.sidebar:
    st.markdown("### 🎮 Menú de Aventuras")
    pagina = st.radio(
        "¿A dónde quieres ir?",
        ["🏠 Inicio", "🔍 Diagnóstico Mágico", "📚 Aprende Jugando", "💊 Guía para Papás", "🏆 Mis Logros"]
    )
    
    # Información del sistema
    with st.expander("ℹ️ Información del Sistema"):
        st.write(f"🆔 Sesión: {st.session_state.session_id[:12]}...")
        st.write(f"💾 Base de datos: Conectada")
        st.write(f"🌐 Entorno: Streamlit Cloud")
        
        # Estadísticas generales (cached)
        @st.cache_data(ttl=300)  # Cache por 5 minutos
        def get_stats():
            try:
                cursor = st.session_state.db.cursor()
                cursor.execute("SELECT COUNT(DISTINCT session_id) FROM usuarios")
                sessions = cursor.fetchone()[0] if cursor.fetchone() else 0
                cursor.execute("SELECT COUNT(*) FROM diagnosticos")
                diagnosticos = cursor.fetchone()[0] if cursor.fetchone() else 0
                return sessions, diagnosticos
            except:
                return 0, 0
        
        sessions, diagnosticos = get_stats()
        st.metric("👥 Sesiones", sessions)
        st.metric("🔍 Diagnósticos", diagnosticos)

# PÁGINA DE INICIO
if pagina == "🏠 Inicio":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ¡Hola pequeño explorador! 👋")
        st.markdown("""
        Soy **Doctor Oso** 🐻 y estoy aquí para ayudarte a entender por qué a veces tu naricita 
        estornuda mucho o tus ojitos se ponen rojos. ¡Juntos vamos a descubrir si tienes 
        **rinitis alérgica**!
        
        ### 🎯 ¿Qué podemos hacer juntos?
        - 🔍 **Diagnóstico Mágico**: Responde preguntas divertidas
        - 📚 **Aprende Jugando**: Descubre qué es la rinitis alérgica  
        - 💊 **Guía para Papás**: Información para tus padres
        - 🏆 **Mis Logros**: Colecciona insignias geniales
        """)
    
    with col2:
        st.markdown("### 🐻 Doctor Oso")
        st.markdown("""
        ```
            ʕ•ᴥ•ʔ
           /     \\
          |  👩‍⚕️  |
          |  🩺   |
           \\     /
            -----
        ```
        """)
    
    # Manejo de usuario actual
    if st.session_state.usuario_actual:
        st.success(f"¡Hola de nuevo, {st.session_state.usuario_actual['nombre']}! 🎉")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👤 Usuario", st.session_state.usuario_actual['nombre'])
        with col2:
            st.metric("🎂 Edad", f"{st.session_state.usuario_actual['edad']} años")
        with col3:
            logros_count = len(obtener_logros_usuario(st.session_state.usuario_actual['id']))
            st.metric("🏆 Logros", logros_count)
        
        if st.button("🔄 Cambiar de usuario", use_container_width=True):
            st.session_state.usuario_actual = None
            st.rerun()
    else:
        # Registro de usuario
        st.markdown("### 🎪 ¡Regístrate para la aventura!")
        
        with st.form("registro_form", clear_on_submit=False):
            st.markdown("**Completa tus datos para comenzar:**")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                nombre = st.text_input(
                    "🌟 ¿Cómo te llamas?", 
                    placeholder="Ej: María, Carlos, Ana...",
                    max_chars=30,
                    help="Escribe tu nombre (mínimo 2 letras)"
                )
            with col2:
                edad = st.selectbox(
                    "🎂 ¿Cuántos años tienes?", 
                    options=[6, 7, 8, 9, 10],
                    index=2,
                    help="Selecciona tu edad"
                )
            
            st.markdown("🚀 *¡Estás a un clic de comenzar tu aventura médica!*")
            
            submitted = st.form_submit_button(
                "🎉 ¡COMENZAR MI AVENTURA!", 
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                if not nombre or len(nombre.strip()) < 2:
                    st.error("🚫 Por favor, escribe un nombre válido (mínimo 2 letras).")
                elif not nombre.replace(" ", "").isalpha():
                    st.error("🚫 Tu nombre solo puede contener letras.")
                else:
                    with st.spinner("🔄 Creando tu perfil de aventurero..."):
                        try:
                            nombre_limpio = nombre.strip().title()
                            usuario_id = registrar_usuario(nombre_limpio, edad)
                            
                            if usuario_id:
                                st.session_state.usuario_actual = {
                                    'id': usuario_id,
                                    'nombre': nombre_limpio,
                                    'edad': edad
                                }
                                
                                agregar_logro(usuario_id, "Primera Aventura")
                                
                                st.success(f"🎉 ¡Bienvenido {nombre_limpio}!")
                                st.info("✨ Has obtenido tu primera insignia: **Primera Aventura**")
                                st.balloons()
                                
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error("❌ Hubo un problema al crear tu perfil. Intenta de nuevo.")
                        except Exception as e:
                            st.error(f"❌ Error inesperado: {str(e)}")

# PÁGINA DE DIAGNÓSTICO
elif pagina == "🔍 Diagnóstico Mágico":
    if not st.session_state.usuario_actual:
        st.warning("¡Primero regístrate en la página de Inicio! 🏠")
        st.stop()
    
    st.markdown(f"### 🔮 ¡Hola {st.session_state.usuario_actual['nombre']}! Vamos a descubrir juntos")
    
    st.markdown("""
    <div class="symptom-card">
    <h4>🎯 Voy a hacerte algunas preguntas mágicas sobre cómo te sientes</h4>
    <p>Responde con honestidad, ¡no hay respuestas incorrectas!</p>
    </div>
    """, unsafe_allow_html=True)
    
    sintomas_seleccionados = []
    
    # Usar formulario para mejor UX
    with st.form("diagnostico_form"):
        st.markdown("#### 🤧 Sobre tu naricita y estornudos:")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox("🤧 ¿Estornudas mucho durante el día?"):
                sintomas_seleccionados.append('estornudos_frecuentes')
            if st.checkbox("👃 ¿Te pica mucho la nariz?"):
                sintomas_seleccionados.append('picazon_nariz')
            if st.checkbox("🚫 ¿Se te tapa la nariz frecuentemente?"):
                sintomas_seleccionados.append('congestion_nasal')
        
        with col2:
            if st.checkbox("💧 ¿Te gotea mucho la nariz?"):
                sintomas_seleccionados.append('goteo_nasal')
            if st.checkbox("👀 ¿Se te ponen rojos o llorosos los ojos?"):
                sintomas_seleccionados.append('ojos_rojos_llorosos')
            if st.checkbox("😴 ¿Te sientes más cansado de lo normal?"):
                sintomas_seleccionados.append('fatiga')
        
        st.markdown("#### 🌸 Sobre cuándo te sientes peor:")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox("🌸 ¿Te sientes peor en primavera?"):
                sintomas_seleccionados.append('empeora_primavera')
            if st.checkbox("🐕 ¿Estornudas más cerca de mascotas?"):
                sintomas_seleccionados.append('reaccion_mascotas')
        
        with col2:
            if st.checkbox("🧹 ¿Te molesta el polvo en casa?"):
                sintomas_seleccionados.append('reaccion_polvo')
        
        # Botón de diagnóstico
        submitted = st.form_submit_button("🔮 ¡Hacer mi diagnóstico mágico!", type="primary", use_container_width=True)
        
        if submitted:
            if sintomas_seleccionados:
                with st.spinner("🔮 Analizando tus síntomas mágicos..."):
                    resultado = st.session_state.sistema_experto.diagnosticar(sintomas_seleccionados)
                    
                    # Guardar diagnóstico
                    guardar_diagnostico(
                        st.session_state.usuario_actual['id'], 
                        sintomas_seleccionados, 
                        resultado['probabilidad']
                    )
                    
                    # Mostrar resultado
                    st.markdown(f"""
                    <div class="diagnosis-result">
                        <h3>🎊 ¡Resultado de tu diagnóstico mágico!</h3>
                        <h2>{resultado['interpretacion']}</h2>
                        <p><strong>Probabilidad: {resultado['probabilidad']:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Consejos según resultado
                    if resultado['probabilidad'] >= 40:
                        st.warning("""
                        ### 👨‍⚕️ ¡Importante para tus papás!
                        Los síntomas que describes podrían indicar rinitis alérgica. 
                        Es recomendable que tus papás te lleven con un doctor para 
                        confirmar el diagnóstico y recibir el mejor tratamiento.
                        """)
                        agregar_logro(st.session_state.usuario_actual['id'], "Detective de Síntomas")
                    else:
                        st.success("""
                        ### 🌟 ¡Qué bueno!
                        No pareces tener muchos síntomas de rinitis alérgica, 
                        pero si te sientes mal, siempre es bueno hablar con un doctor.
                        """)
                    
                    st.balloons()
                    agregar_logro(st.session_state.usuario_actual['id'], "Primera Consulta")
            else:
                st.error("¡Necesitas seleccionar al menos un síntoma para hacer el diagnóstico! 😊")

# PÁGINA EDUCATIVA
elif pagina == "📚 Aprende Jugando":
    st.markdown("### 🎓 ¡Vamos a aprender sobre la rinitis alérgica!")
    
    tab1, tab2, tab3 = st.tabs(["🤔 ¿Qué es?", "🗺️ Mapa de Alérgenos", "🎯 Misiones de Prevención"])
    
    with tab1:
        st.markdown("#### 🌟 ¿Qué es la Rinitis Alérgica?")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            Imagínate que tu naricita es como un **guardián súper protector** 🛡️. 
            Su trabajo es cuidarte de cosas que podrían hacerte daño.
            
            Pero a veces, este guardián se confunde y piensa que cosas inofensivas 
            como el **polen de las flores** 🌸 o el **pelo de los gatitos** 🐱 
            son enemigos peligrosos.
            
            ¡Entonces tu nariz empieza a estornudar y tus ojos se ponen llorosos 
            para "expulsar" a estos "invasores"! Pero en realidad, no son peligrosos.
            
            A esto le llamamos **rinitis alérgica** - cuando tu nariz reacciona 
            demasiado fuerte a cosas que normalmente no deberían molestarte.
            """)
        
        with col2:
            st.markdown("""
            ### 🤧 Síntomas comunes:
            - Estornudos 🤧
            - Picazón en nariz 👃
            - Ojos llorosos 👀💧
            - Congestión nasal 🚫
            """)
        
        if st.button("🏆 ¡Ya entendí qué es!"):
            if st.session_state.usuario_actual:
                if agregar_logro(st.session_state.usuario_actual['id'], "Pequeño Científico"):
                    st.success("¡Felicidades! Obtuviste la insignia de 'Pequeño Científico' 🏆")
                    st.balloons()
    
    with tab2:
        st.markdown("#### 🗺️ Mapa de los Alérgenos Misteriosos")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="mission-card">
            <h5>🌸 Polen (Primavera)</h5>
            <p>Las flores sueltan polvito amarillo que vuela por el aire</p>
            <p><strong>Dónde:</strong> Parques, jardines</p>
            <p><strong>Cuándo:</strong> Primavera y verano</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="mission-card">
            <h5>🕷️ Ácaros del Polvo</h5>
            <p>Bichitos súper pequeños que viven en el polvo</p>
            <p><strong>Dónde:</strong> Camas, alfombras, cortinas</p>
            <p><strong>Cuándo:</strong> Todo el año</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="mission-card">
            <h5>🐕🐱 Mascotas</h5>
            <p>Pelitos y caspa de perros y gatos</p>
            <p><strong>Dónde:</strong> Casas con mascotas</p>
            <p><strong>Cuándo:</strong> Todo el año</p>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🕵️ ¡Ya conozco a los alérgenos!"):
            if st.session_state.usuario_actual:
                if agregar_logro(st.session_state.usuario_actual['id'], "Explorador de Alérgenos"):
                    st.success("¡Genial! Obtuviste la insignia de 'Explorador de Alérgenos' 🕵️")
                    st.balloons()
    
    with tab3:
        st.markdown("#### 🎯 Misiones Especiales de Prevención")
        
        misiones = [
            {"titulo": "🧹 Misión: Habitación Súper Limpia", 
             "descripcion": "Mantén tu cuarto libre de polvo aspirando y limpiando seguido",
             "puntos": "⭐⭐⭐"},
            {"titulo": "🤲 Mis
