import streamlit as st
import random
import time

# --- Constantes del Juego ---
NUM_FILAS = 9
NUM_COLUMNAS = 9
NUM_MINAS = 10

# --- Emojis para la UI ---
EMOJIS = {
    "mina": "💣",
    "bandera": "🚩",
    "oculto": "⬜",
    "vacio": "🟦",
    0: "0️⃣", 1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣",
    5: "5️⃣", 6: "6️⃣", 7: "7️⃣", 8: "8️⃣"
}

# --- Funciones de Inicialización del Tablero ---

def generar_tablero_minas(filas, columnas, minas, primera_fila, primera_columna):
    """Genera un tablero con minas, asegurando que la primera casilla no sea una mina."""
    tablero = [['' for _ in range(columnas)] for _ in range(filas)]
    minas_colocadas = 0

    while minas_colocadas < minas:
        r = random.randint(0, filas - 1)
        c = random.randint(0, columnas - 1)
        
        # Asegurarse de no colocar una mina en la primera casilla clicada ni sus vecinos
        if (r == primera_fila and c == primera_columna) or \
           abs(r - primera_fila) <= 1 and abs(c - primera_columna) <= 1:
            continue
            
        if tablero[r][c] != 'M':
            tablero[r][c] = 'M'
            minas_colocadas += 1
    return tablero

def calcular_numeros(tablero):
    """Calcula el número de minas adyacentes para cada celda no minada."""
    filas = len(tablero)
    columnas = len(tablero[0])

    for r in range(filas):
        for c in range(columnas):
            if tablero[r][c] == 'M':
                continue
            
            minas_adyacentes = 0
            # Revisar las 8 celdas vecinas
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue # Es la celda actual

                    nr, nc = r + dr, c + dc
                    if 0 <= nr < filas and 0 <= nc < columnas and tablero[nr][nc] == 'M':
                        minas_adyacentes += 1
            tablero[r][c] = minas_adyacentes
    return tablero

def inicializar_juego():
    """Inicializa todas las variables de estado del juego."""
    st.session_state.tablero_visible = [[EMOJIS["oculto"] for _ in range(NUM_COLUMNAS)] for _ in range(NUM_FILAS)]
    st.session_state.tablero_interno = None # Se generará en el primer clic
    st.session_state.juego_terminado = False
    st.session_state.resultado = ""
    st.session_state.primer_clic = True
    st.session_state.minas_restantes = NUM_MINAS
    st.session_state.celdas_reveladas = 0

# --- Lógica del Juego ---

def revelar_celda(r, c):
    """Revela una celda y propaga si es una celda vacía (0 minas)."""
    if st.session_state.juego_terminado:
        return

    # Primer clic: generar tablero interno
    if st.session_state.primer_clic:
        st.session_state.tablero_interno = generar_tablero_minas(NUM_FILAS, NUM_COLUMNAS, NUM_MINAS, r, c)
        st.session_state.tablero_interno = calcular_numeros(st.session_state.tablero_interno)
        st.session_state.primer_clic = False

    # No permitir revelar si ya está revelada o es una bandera
    if st.session_state.tablero_visible[r][c] != EMOJIS["oculto"]:
        return

    # Si es una mina, fin del juego
    if st.session_state.tablero_interno[r][c] == 'M':
        st.session_state.tablero_visible[r][c] = EMOJIS["mina"]
        st.session_state.juego_terminado = True
        st.session_state.resultado = "¡BOOM! 💥 Has tocado una mina. ¡Juego Terminado!"
        revelar_todas_minas()
        return

    # Si es una celda con número
    if isinstance(st.session_state.tablero_interno[r][c], int) and st.session_state.tablero_interno[r][c] > 0:
        st.session_state.tablero_visible[r][c] = EMOjis[st.session_state.tablero_interno[r][c]]
        st.session_state.celdas_reveladas += 1
        return

    # Si es una celda vacía (0 minas), propagar
    if st.session_state.tablero_interno[r][c] == 0:
        propagar_vacio(r, c)
    
    verificar_victoria()

def propagar_vacio(r_inicio, c_inicio):
    """
    Algoritmo recursivo para revelar celdas vacías y sus vecinas.
    """
    pila = [(r_inicio, c_inicio)]
    visitadas = set()

    while pila:
        r, c = pila.pop()

        if (r, c) in visitadas:
            continue
        visitadas.add((r, c))

        # Si ya está visible (no oculta y no bandera), no hacer nada
        if st.session_state.tablero_visible[r][c] != EMOJIS["oculto"]:
            continue
        
        # Si es una bandera, no revelar
        if st.session_state.tablero_visible[r][c] == EMOJIS["bandera"]:
            continue

        # Revelar celda actual
        if st.session_state.tablero_interno[r][c] == 'M':
            continue # No debería pasar, pero por seguridad
        elif isinstance(st.session_state.tablero_interno[r][c], int):
            st.session_state.tablero_visible[r][c] = EMOJIS[st.session_state.tablero_interno[r][c]]
            st.session_state.celdas_reveladas += 1
        
        # Si la celda actual es 0, añadir vecinos a la pila
        if st.session_state.tablero_interno[r][c] == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < NUM_FILAS and 0 <= nc < NUM_COLUMNAS:
                        if (nr, nc) not in visitadas and \
                           st.session_state.tablero_visible[nr][nc] == EMOJIS["oculto"]:
                            pila.append((nr, nc))

def marcar_bandera(r, c):
    """Coloca o quita una bandera en una celda."""
    if st.session_state.juego_terminado or st.session_state.primer_clic:
        return

    current_state = st.session_state.tablero_visible[r][c]

    if current_state == EMOJIS["oculto"]:
        if st.session_state.minas_restantes > 0:
            st.session_state.tablero_visible[r][c] = EMOJIS["bandera"]
            st.session_state.minas_restantes -= 1
    elif current_state == EMOJIS["bandera"]:
        st.session_state.tablero_visible[r][c] = EMOJIS["oculto"]
        st.session_state.minas_restantes += 1
    
    verificar_victoria()

def revelar_todas_minas():
    """Muestra todas las minas al final del juego."""
    if st.session_state.tablero_interno:
        for r in range(NUM_FILAS):
            for c in range(NUM_COLUMNAS):
                if st.session_state.tablero_interno[r][c] == 'M' and \
                   st.session_state.tablero_visible[r][c] != EMOJIS["bandera"]:
                    st.session_state.tablero_visible[r][c] = EMOJIS["mina"]

def verificar_victoria():
    """Comprueba si el jugador ha ganado el juego."""
    # El jugador gana si todas las celdas no minadas han sido reveladas.
    # Opcional: si todas las minas están correctamente marcadas con banderas.
    
    celdas_no_minadas = (NUM_FILAS * NUM_COLUMNAS) - NUM_MINAS

    if st.session_state.celdas_reveladas == celdas_no_minadas and \
       not st.session_state.juego_terminado: # Asegurarse de que no haya explotado una mina
        st.session_state.juego_terminado = True
        st.session_state.resultado = "🎉 ¡Felicidades! ¡Has ganado el Buscaminas!"

# --- Interfaz de Streamlit ---

st.set_page_config(layout="wide")
st.title("💣 Buscaminas")
st.markdown("---")

if 'tablero_visible' not in st.session_state:
    inicializar_juego()

col_left, col_game, col_right = st.columns([1, 2, 1])

with col_game:
    st.markdown(f"### Minas Restantes: {st.session_state.minas_restantes} {EMOJIS['bandera']}")
    
    if st.session_state.juego_terminado:
        if "BOOM" in st.session_state.resultado:
            st.error(st.session_state.resultado)
        else:
            st.success(st.session_state.resultado)
    else:
        st.info("Haz clic para revelar, o usa 'Marcar Bandera' y luego clic para marcar/desmarcar.")

    modo_bandera = st.checkbox("🚩 Marcar Bandera", key="modo_bandera")

    # Mostrar el tablero
    tablero_display = ""
    for r in range(NUM_FILAS):
        cols = st.columns(NUM_COLUMNAS)
        for c in range(NUM_COLUMNAS):
            with cols[c]:
                # Crear un botón para cada celda
                key_button = f"btn_{r}_{c}"
                
                # Deshabilitar botones si el juego ha terminado, a menos que sean minas para visualización
                disabled = st.session_state.juego_terminado and \
                           st.session_state.tablero_visible[r][c] != EMOJIS["mina"] and \
                           st.session_state.tablero_visible[r][c] != EMOJIS["bandera"]


                if st.button(st.session_state.tablero_visible[r][c], key=key_button, use_container_width=True, disabled=disabled):
                    if not st.session_state.juego_terminado:
                        if modo_bandera:
                            marcar_bandera(r, c)
                        else:
                            revelar_celda(r, c)
    
    st.markdown("---")
    if st.button("🔄 Reiniciar Juego", type="primary", use_container_width=True):
        inicializar_juego()
        st.experimental_rerun() # Fuerza una recarga para limpiar el estado de los botones