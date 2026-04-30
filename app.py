import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime

# --- CONFIGURACIÓN DE BASE DE DATOS ---
def init_db():
    conn = sqlite3.connect('anzuelo_data.db')
    c = conn.cursor()
    # Tabla de Proveedores
    c.execute('CREATE TABLE IF NOT EXISTS proveedores (id INTEGER PRIMARY KEY, nombre TEXT UNIQUE)')
    # Tabla de Compras (Basada en tus imágenes de mayorista y bebidas)
    c.execute('''CREATE TABLE IF NOT EXISTS compras (
                    id INTEGER PRIMARY KEY, 
                    fecha DATE, 
                    producto TEXT, 
                    categoria TEXT, 
                    cantidad REAL, 
                    formato TEXT, 
                    precio_u REAL, 
                    total REAL, 
                    proveedor TEXT)''')
    conn.commit()
    conn.close()

# --- FUNCIONES DE AYUDA ---
def guardar_compra(fecha, producto, cat, cant, form, precio, prov):
    conn = sqlite3.connect('anzuelo_data.db')
    c = conn.cursor()
    total = cant * precio
    c.execute('''INSERT INTO compras (fecha, producto, categoria, cantidad, formato, precio_u, total, proveedor) 
                 VALUES (?,?,?,?,?,?,?,?)''', (fecha, producto, cat, cant, form, precio, total, prov))
    conn.commit()
    conn.close()

# --- INTERFAZ STREAMLIT ---
def main():
    st.set_page_config(page_title="El Anzuelo - Gestión", layout="wide")
    init_db()

    st.sidebar.title("🔱 Menú Principal")
    menu = ["Dashboard / Reportes", "Registro de Compras", "Gestión de Proveedores"]
    choice = st.sidebar.radio("Seleccione una opción", menu)

    if choice == "Gestión de Proveedores":
        st.header("🤝 Registro de Proveedores")
        nuevo_p = st.text_input("Nombre del nuevo proveedor")
        if st.button("Añadir Proveedor"):
            conn = sqlite3.connect('anzuelo_data.db')
            try:
                conn.execute('INSERT INTO proveedores (nombre) VALUES (?)', (nuevo_p,))
                conn.commit()
                st.success("Proveedor guardado.")
            except:
                st.error("El proveedor ya existe.")
            conn.close()

    elif choice == "Registro de Compras":
        st.header("📝 Ingreso de Mercadería (Bebidas y Mayorista)")
        
        # Obtener proveedores para el selectbox
        conn = sqlite3.connect('anzuelo_data.db')
        prov_df = pd.read_sql("SELECT nombre FROM proveedores", conn)
        conn.close()

        with st.form("form_compras"):
            col1, col2 = st.columns(2)
            fecha = col1.date_input("Fecha de Compra", datetime.now())
            producto = col1.text_input("Producto (Ej: Arroz, Cerveza, etc.)")
            categoria = col1.selectbox("Categoría", ["ABARROTES", "BEBIDAS", "LIMPIEZA", "PLASTICO", "MENESTRA"])
            proveedor = col1.selectbox("Proveedor", prov_df['nombre'].tolist() if not prov_df.empty else ["Debe registrar un proveedor"])
            
            cantidad = col2.number_input("Cantidad", min_value=0.0, step=0.1)
            formato = col2.text_input("Formato (Ej: SACOX50K, CAJAX12, UNIDAD)")
            precio_u = col2.number_input("Precio Unitario (S/.)", min_value=0.0)
            
            if st.form_submit_button("Guardar Registro"):
                guardar_compra(fecha, producto, categoria, cantidad, formato, precio_u, proveedor)
                st.success(f"✅ {producto} registrado correctamente.")

    elif choice == "Dashboard / Reportes":
        st.header("📈 Reporte de Gastos e Inversión")
        conn = sqlite3.connect('anzuelo_data.db')
        df = pd.read_sql("SELECT * FROM compras", conn)
        conn.close()

        if not df.empty:
            df['fecha'] = pd.to_datetime(df['fecha'])
            
            # KPIs
            total_invertido = df['total'].sum()
            st.metric("Inversión Total Acumulada", f"S/. {total_invertido:,.2f}")

            c1, c2 = st.columns(2)
            with c1:
                fig_pie = px.pie(df, values='total', names='categoria', title="Gasto por Categoría")
                st.plotly_chart(fig_pie)
            with c2:
                fig_bar = px.bar(df, x='fecha', y='total', color='categoria', title="Historial de Compras")
                st.plotly_chart(fig_bar)
            
            st.subheader("Detalle Completo")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Aún no hay datos registrados.")

if __name__ == '__main__':
    main()