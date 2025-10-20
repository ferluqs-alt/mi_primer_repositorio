import streamlit as st
import random
import time

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Rifa Online con Streamlit",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Inicialización del Estado de Sesión ---
def inicializar_participantes():
    """Inicializa la lista de participantes en el estado de sesión."""
    if 'participantes' not in st.session_state:
        st.session_state.participantes = []
    if 'ganadores' not in st.session_state:
        st.session_state.ganadores = []

# --- Lógica del Sorteo ---
def realizar_sorteo(participantes, num_ganadores):
    """Selecciona los ganadores de forma aleatoria."""
    if not participantes:
        st.error("No hay participantes registrados para realizar el sorteo.")
        return
        
    # Asegurarse de que el número de ganadores no exceda el número de participantes
    num_ganadores = min(num_ganadores, len(participantes))
    
    # Usar random.sample para seleccionar ganadores sin reemplazo
    ganadores_seleccionados = random.sample(participantes, num_ganadores)
    
    # Guardar los ganadores en el estado de sesión
    st.session_state.ganadores = ganadores_seleccionados
    
    return ganadores_seleccionados

def agregar_participante(nombre):
    """Añade un nuevo participante a la lista si el nombre no está vacío."""
    nombre = nombre.strip()
    if nombre and nombre not in st.session_state.participantes:
        st.session_state.participantes.append(nombre)
        st.success(f"🎉 ¡{nombre} añadido con éxito!")
    elif nombre in st.session_state.participantes:
        st.warning("Ese participante ya está en la lista.")

def limpiar_lista():
    """Reinicia la lista de participantes y ganadores."""
    st.session_state.participantes = []
    st.session_state.ganadores = []
    st.info("Lista de participantes reiniciada.")

# --- Interfaz de Streamlit ---
inicializar_participantes()

st.title("🎁 Rifa Online Automática")
st.markdown("---")

# 1. Zona de Registro de Participantes
st.header("1. Registrar Participantes")
with st.form(key='registro_form', clear_on_submit=True):
    # Campo de entrada para el nuevo participante
    nuevo_participante = st.text_input("Nombre o número del participante:", key="input_participante")
    
    col_add, col_clean = st.columns([1, 1])
    with col_add:
        submit_button = st.form_submit_button(label='➕ Añadir a la Rifa')
    with col_clean:
        if st.form_submit_button(label='🗑️ Limpiar Lista'):
            limpiar_lista()
            st.experimental_rerun() # Fuerza la recarga para actualizar

if submit_button and nuevo_participante:
    agregar_participante(nuevo_participante)

st.markdown("---")

# 2. Lista de Participantes Actual
st.header("2. Lista Actual")

if st.session_state.participantes:
    st.info(f"Total de participantes: **{len(st.session_state.participantes)}**")
    # Mostrar la lista en formato numerado
    lista_str = "\n".join([f"**{i+1}.** {p}" for i, p in enumerate(st.session_state.participantes)])
    st.markdown(lista_str)
else:
    st.warning("Aún no hay participantes registrados.")

st.markdown("---")

# 3. Ejecutar Sorteo
st.header("3. Ejecutar Sorteo")

if len(st.session_state.participantes) > 0:
    # Selector de cuántos ganadores
    max_ganadores = len(st.session_state.participantes)
    num_ganadores = st.slider(
        "Número de Ganadores a Sortear:",
        min_value=1,
        max_value=max_ganadores,
        value=min(1, max_ganadores), # Valor por defecto 1 o el máximo si hay menos
        step=1
    )

    if st.button(f"🎉 ¡REALIZAR SORTEO DE {num_ganadores} GANADOR(ES)! 🎉", type="primary", use_container_width=True):
        st.session_state.ganadores = realizar_sorteo(st.session_state.participantes, num_ganadores)

        if st.session_state.ganadores:
            st.balloons() # Animación de celebración

            st.subheader("¡Los Ganadores son:")
            
            # Mostrar ganadores con efecto de retraso (simula tensión)
            ganadores_emoji = ["🥇", "🥈", "🥉"] + ["🏅"] * (len(st.session_state.ganadores) - 3)
            
            # Iterar para mostrar uno por uno con un pequeño delay
            for i, ganador in enumerate(st.session_state.ganadores):
                placeholder = st.empty()
                placeholder.metric(label=f"Posición {i+1}", value=f"❓❓❓")
                time.sleep(0.5) # Pausa para el efecto
                placeholder.metric(label=f"{ganadores_emoji[i]} Ganador #{i+1}", value=ganador)
                
            st.success("¡Sorteo completado con éxito! ¡Felicidades!")
else:
    st.error("Necesitas al menos un participante para ejecutar el sorteo.")

st.markdown("---")