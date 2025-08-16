import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import sys
import os

# Configuración de la página
st.set_page_config(
    page_title="🐻 Doctor Solución",
    page_icon="🐻",
    layout="wide",
    initial_sidebar_state="expanded"
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
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Funciones para base de datos
def get_db_path():
    """Obtiene la ruta correcta para la base de datos"""
    # Para Streamlit Cloud, usar directorio temporal
    return 'doctor_solucion.db'

# Inicializar base de datos
@st.cache_resource
def init_db():
    """Inicializa la base de datos con conexión thread-safe"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()

    # Tabla para usuarios
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY,
            nombre TEXT,
            edad INTEGER,
            fecha_registro DATE
        )
    ''')

    # Tabla para diagnósticos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnosticos (
            id INTEGER PRIMARY KEY,
            usuario_id INTEGER,
            sintomas TEXT,
            probabilidad REAL,
            fecha DATE,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')

    # Tabla para logros
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logros (
            id INTEGER PRIMARY KEY,
            usuario_id INTEGER,
            tipo_logro TEXT,
            fecha DATE,
            FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
        )
    ''')

    conn.commit()
    return conn

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

# Inicializar sistema
if 'db' not in st.session_state:
    st.session_state.db = init_db()
if 'sistema_experto' not in st.session_state:
    st.session_state.sistema_experto = SistemaExperto()
if 'usuario_actual' not in st.session_state:
    st.session_state.usuario_actual = None
if 'logros' not in st.session_state:
    st.session_state.logros = []

# Funciones auxiliares
def registrar_usuario(nombre, edad):
    cursor = st.session_state.db.cursor()
    cursor.execute(
        "INSERT INTO usuarios (nombre, edad, fecha_registro) VALUES (?, ?, ?)",
        (nombre, edad, datetime.now().date())
    )
    st.session_state.db.commit()
    return cursor.lastrowid

def obtener_logros_usuario(usuario_id):
    cursor = st.session_state.db.cursor()
    cursor.execute("SELECT tipo_logro FROM logros WHERE usuario_id = ?", (usuario_id,))
    return [row[0] for row in cursor.fetchall()]

def agregar_logro(usuario_id, tipo_logro):
    # Verificar si ya tiene este logro
    logros_existentes = obtener_logros_usuario(usuario_id)
    if tipo_logro not in logros_existentes:
        cursor = st.session_state.db.cursor()
        cursor.execute(
            "INSERT INTO logros (usuario_id, tipo_logro, fecha) VALUES (?, ?, ?)",
            (usuario_id, tipo_logro, datetime.now().date())
        )
        st.session_state.db.commit()
        return True
    return False

# Información específica para Streamlit Cloud
st.markdown("""
<div class="streamlit-info">
<h4>☁️ Ejecutándose en Streamlit Cloud</h4>
<p>¡Bienvenido a Doctor Solución! Esta aplicación funciona perfectamente en Streamlit Cloud.
Los datos se almacenan temporalmente durante tu sesión.</p>
</div>
""", unsafe_allow_html=True)

# INTERFAZ PRINCIPAL
st.title("🐻 ¡Hola! Soy Doctor Oso, tu amigo médico")
st.markdown("### 🌟 Bienvenido a Doctor Solución - Tu aventura médica comienza aquí")

# Sidebar para navegación
with st.sidebar:
    st.markdown("### 🎮 Menú de Aventuras")
    pagina = st.radio(
        "¿A dónde quieres ir?",
        ["🏠 Inicio", "🔍 Diagnóstico Mágico", "📚 Aprende Jugando", "💊 Guía para Papás", "🏆 Mis Logros"]
    )

    # Información técnica
    with st.expander("🔧 Info Técnica"):
        st.write(f"📁 BD: {get_db_path()}")
        st.write(f"🟢 Estado: Conectado")
        if st.button("🔄 Reiniciar BD"):
            st.cache_resource.clear()
            st.session_state.db = init_db()
            st.success("✅ Base de datos reiniciada")

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

    # Registro de usuario
    with st.expander("🎪 ¡Regístrate para la aventura!", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            nombre = st.text_input("🌟 ¿Cómo te llamas?", placeholder="Escribe tu nombre aquí")
        with col2:
            edad = st.number_input("🎂 ¿Cuántos años tienes?", min_value=6, max_value=10, value=8)

        if st.button("🚀 ¡Comenzar mi aventura!"):
            if nombre:
                usuario_id = registrar_usuario(nombre, edad)
                st.session_state.usuario_actual = {'id': usuario_id, 'nombre': nombre, 'edad': edad}
                if agregar_logro(usuario_id, "Primera Aventura"):
                    st.success(f"¡Bienvenido {nombre}! 🎉 Ya puedes comenzar tu aventura médica.")
                    st.balloons()
            else:
                st.error("¡Necesito saber tu nombre para ser tu amigo médico! 😊")

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

    # Preguntas sobre síntomas principales
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

    # Preguntas sobre contexto/factores desencadenantes
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
    if st.button("🔮 ¡Hacer mi diagnóstico mágico!", type="primary"):
        if sintomas_seleccionados:
            resultado = st.session_state.sistema_experto.diagnosticar(sintomas_seleccionados)

            # Guardar diagnóstico
            cursor = st.session_state.db.cursor()
            cursor.execute(
                "INSERT INTO diagnosticos (usuario_id, sintomas, probabilidad, fecha) VALUES (?, ?, ?, ?)",
                (st.session_state.usuario_actual['id'],
                 json.dumps(sintomas_seleccionados),
                 resultado['probabilidad'],
                 datetime.now().date())
            )
            st.session_state.db.commit()

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

    with tab3:
        st.markdown("#### 🎯 Misiones Especiales de Prevención")
        
        missions = [
            {
                "titulo": "🏠 Misión: Casa Súper Limpia",
                "descripcion": "Ayuda a mantener tu casa libre de polvo y alérgenos",
                "tasks": [
                    "Aspira alfombras y tapetes regularmente",
                    "Lava las sábanas con agua caliente",
                    "Ventila tu habitación todos los días"
                ]
            },
            {
                "titulo": "🌸 Misión: Detective del Polen",
                "descripcion": "Aprende cuándo hay más polen en el aire",
                "tasks": [
                    "Revisa el pronóstico de polen",
                    "Evita salir en días muy ventosos",
                    "Cierra las ventanas en primavera"
                ]
            },
            {
                "titulo": "💧 Misión: Guardián Nasal",
                "descripcion": "Cuida tu nariz para que esté siempre sana",
                "tasks": [
                    "Lávate las manos frecuentemente",
                    "No te toques la nariz con las manos sucias",
                    "Usa pañuelos desechables"
                ]
            }
        ]

        for i, mission in enumerate(missions):
            with st.expander(mission["titulo"]):
                st.write(mission["descripcion"])
                for task in mission["tasks"]:
                    st.write(f"✅ {task}")
                
                if st.button(f"¡Completé la misión {i+1}!", key=f"mission_{i}"):
                    if st.session_state.usuario_actual:
                        logro_nombre = f"Misión {i+1} Completada"
                        if agregar_logro(st.session_state.usuario_actual['id'], logro_nombre):
                            st.success(f"¡Felicidades! Completaste la {mission['titulo']} 🎉")

# PÁGINA GUÍA PARA PAPÁS
elif pagina == "💊 Guía para Papás":
    st.markdown("### 👨‍⚕️👩‍⚕️ Guía para Padres: Rinitis Alérgica Infantil")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Información Médica", "🔍 Cuándo Consultar", "💊 Tratamientos", "🏠 Prevención"])
    
    with tab1:
        st.markdown("""
        #### ¿Qué es la Rinitis Alérgica?
        
        La rinitis alérgica es una reacción inflamatoria de la mucosa nasal causada por la exposición 
        a alérgenos ambientales. Es una de las enfermedades crónicas más comunes en la infancia.
        
        **Síntomas principales:**
        - Estornudos repetidos
        - Rinorrea (secreción nasal acuosa)
        - Congestión nasal
        - Prurito (picazón) nasal y ocular
        - Lagrimeo y enrojecimiento ocular
        
        **Factores de riesgo:**
        - Antecedentes familiares de alergias
        - Exposición temprana a alérgenos
        - Otras condiciones alérgicas (asma, dermatitis atópica)
        """)
    
    with tab2:
        st.markdown("""
        #### 🚨 Cuándo Consultar al Médico
        
        **Consulta urgente si:**
        - Dificultad para respirar
        - Sibilancias (silbidos al respirar)
        - Fiebre alta persistente
        - Dolor de oído intenso
        
        **Consulta programada si:**
        - Los síntomas interfieren con el sueño
        - Afecta el rendimiento escolar
        - Los síntomas duran más de 2 semanas
        - No mejora con medidas generales
        
        **Especialistas recomendados:**
        - Pediatra (primera consulta)
        - Alergólogo/Inmunólogo pediátrico
        - Otorrinolaringólogo pediátrico
        """)
    
    with tab3:
        st.markdown("""
        #### 💊 Opciones de Tratamiento
        
        **Medidas no farmacológicas:**
        - Control ambiental (eliminación de alérgenos)
        - Lavado nasal con suero fisiológico
        - Humidificación adecuada del ambiente
        
        **Tratamiento farmacológico (bajo supervisión médica):**
        - Antihistamínicos orales
        - Corticosteroides nasales tópicos
        - Descongestionantes nasales (uso limitado)
        - Cromoglicato sódico nasal
        
        **Inmunoterapia:**
        - Considerar en casos específicos
        - Solo bajo supervisión especializada
        
        ⚠️ **Importante:** Nunca automediques a tu hijo. Siempre consulta con un profesional de la salud.
        """)
    
    with tab4:
        st.markdown("""
        #### 🏠 Medidas de Prevención en Casa
        
        **Control de ácaros del polvo:**
        - Fundas antiácaros en colchones y almohadas
        - Lavado de ropa de cama a >60°C semanalmente
        - Aspirado regular con filtros HEPA
        - Mantener humedad <50%
        
        **Control de polen:**
        - Mantener ventanas cerradas durante temporadas altas
        - Usar aire acondicionado con filtros
        - Evitar actividades al aire libre en días ventosos
        - Ducha después de actividades exteriores
        
        **Control de mascotas:**
        - Mantener mascotas fuera del dormitorio
        - Baño regular de mascotas
        - Uso de filtros de aire HEPA
        
        **Otros consejos:**
        - No fumar en casa
        - Evitar ambientadores fuertes
        - Ventilación adecuada para prevenir moho
        """)

# PÁGINA DE LOGROS
elif pagina == "🏆 Mis Logros":
    if not st.session_state.usuario_actual:
        st.warning("¡Primero regístrate en la página de Inicio! 🏠")
        st.stop()
    
    st.markdown(f"### 🏆 Los Logros de {st.session_state.usuario_actual['nombre']}")
    
    logros_usuario = obtener_logros_usuario(st.session_state.usuario_actual['id'])
    
    # Definir todos los logros posibles
    todos_los_logros = {
        "Primera Aventura": {"emoji": "🌟", "desc": "¡Iniciaste tu primera aventura médica!"},
        "Primera Consulta": {"emoji": "👩‍⚕️", "desc": "¡Hiciste tu primer diagnóstico!"},
        "Detective de Síntomas": {"emoji": "🕵️", "desc": "¡Encontraste síntomas importantes!"},
        "Pequeño Científico": {"emoji": "🧬", "desc": "¡Aprendiste qué es la rinitis alérgica!"},
        "Explorador de Alérgenos": {"emoji": "🗺️", "desc": "¡Conoces los alérgenos misteriosos!"},
        "Misión 1 Completada": {"emoji": "🏠", "desc": "¡Completaste la Misión Casa Súper Limpia!"},
        "Misión 2 Completada": {"emoji": "🌸", "desc": "¡Completaste la Misión Detective del Polen!"},
        "Misión 3 Completada": {"emoji": "💧", "desc": "¡Completaste la Misión Guardián Nasal!"}
    }
    
    if logros_usuario:
        st.markdown("#### 🎉 ¡Mira todas las insignias que has ganado!")
        
        cols = st.columns(3)
        for i, logro in enumerate(logros_usuario):
            with cols[i % 3]:
                if logro in todos_los_logros:
                    st.markdown(f"""
                    <div class="achievement-badge">
                    {todos_los_logros[logro]['emoji']} {logro}
                    <br><small>{todos_los_logros[logro]['desc']}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown(f"### 📊 Estadísticas de {st.session_state.usuario_actual['nombre']}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏆 Total de Logros", len(logros_usuario))
        
        with col2:
            # Contar diagnósticos
            cursor = st.session_state.db.cursor()
            cursor.execute("SELECT COUNT(*) FROM diagnosticos WHERE usuario_id = ?"), 
            st.write(st.session_state)
