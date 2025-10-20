import streamlit as st
import random

# --- Configuración del Juego y Opciones ---
OPCIONES = {
    "Piedra": "✊",
    "Papel": "✋",
    "Tijera": "✌️"
}

# Reglas: (Opción_Jugador, Opción_Computadora) -> Resultado
REGLAS = {
    ("Piedra", "Tijera"): "Ganas",
    ("Tijera", "Papel"): "Ganas",
    ("Papel", "Piedra"): "Ganas",
}

# --- Inicialización del Estado de Sesión ---
def inicializar_estado():
    """Inicializa las variables de estado si no existen."""
    if 'puntuacion_jugador' not in st.session_state:
        st.session_state.puntuacion_jugador = 0
    if 'puntuacion_computadora' not in st.session_state:
        st.session_state.puntuacion_computadora = 0
    if 'mensaje_ronda' not in st.session_state:
        st.session_state.mensaje_ronda = "¡Elige tu movimiento para empezar!"

# --- Lógica del Juego ---
def jugar_ronda(eleccion_jugador):
    """
    Determina el resultado de la ronda y actualiza el estado.
    """
    
    # 1. Elección de la Computadora
    opciones_lista = list(OPCIONES.keys())
    eleccion_computadora = random.choice(opciones_lista)
    
    # 2. Determinar el resultado
    resultado = ""
    if eleccion_jugador == eleccion_computadora:
        resultado = "Empate"
    elif (eleccion_jugador, eleccion_computadora) in REGLAS:
        resultado = "Ganas"
        st.session_state.puntuacion_jugador += 1
    else:
        resultado = "Pierdes"
        st.session_state.puntuacion_computadora += 1
        
    # 3. Actualizar mensaje de la ronda
    jugador_emoji = OPCIONES[eleccion_jugador]
    comp_emoji = OPCIONES[eleccion_computadora]

    mensaje = (
        f"**Tú:** {eleccion_jugador} {jugador_emoji} | "
        f"**Computadora:** {eleccion_computadora} {comp_emoji} \n\n"
        f"**Resultado:** ¡{resultado}! 🎉" if resultado == "Ganas" else f"**Resultado:** ¡{resultado}! 😞" if resultado == "Pierdes" else f"**Resultado:** ¡{resultado}! 🤝"
    )
    st.session_state.mensaje_ronda = mensaje

# --- Interfaz de Streamlit ---

inicializar_estado()

st.title("Yan Ken Po (Piedra, Papel o Tijera)")
st.subheader("Hecho con Python y Streamlit")

# Mostrar Puntuación Actual
st.markdown(
    f"### 🏆 Marcador: Tú {st.session_state.puntuacion_jugador} - {st.session_state.puntuacion_computadora} Computadora"
)

st.markdown("---")

# Botones de Elección
st.write("### Elige tu opción:")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button(f"{OPCIONES['Piedra']} Piedra", use_container_width=True):
        jugar_ronda("Piedra")

with col2:
    if st.button(f"{OPCIONES['Papel']} Papel", use_container_width=True):
        jugar_ronda("Papel")

with col3:
    if st.button(f"{OPCIONES['Tijera']} Tijera", use_container_width=True):
        jugar_ronda("Tijera")

st.markdown("---")

# Mostrar Resultado de la Última Ronda
st.markdown(f"**Última Ronda:**")
st.info(st.session_state.mensaje_ronda)

# --- Botón para Reiniciar el Juego ---
def reiniciar_juego():
    """Función para reiniciar el marcador."""
    st.session_state.puntuacion_jugador = 0
    st.session_state.puntuacion_computadora = 0
    st.session_state.mensaje_ronda = "Juego reiniciado. ¡Elige tu movimiento!"
    
if st.button("🔄 Reiniciar Juego", type="secondary"):
    reiniciar_juego()