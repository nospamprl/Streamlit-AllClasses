import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
import streamlit as st
import time
import os

# Asset class mapping to model files and datasets
asset_options = {
    "Montacargas": {"model": "best_forklift_model.pkl", "dataset": "Scrap Forklifts.xlsx"},
    "Camiones": {"model": "best_truck_model.pkl", "dataset": "Scrap Trucks.xlsx"},
    "Construcción": {"model": "best_construction_model.pkl", "dataset": "Scrap Construction.xlsx"}
}

# Streamlit UI
st.set_page_config(page_title="Predicción de Precios", layout="centered")
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🚜 Predicción de Precios de Equipos")
with col2:
    st.image("logo.png", width=100)
st.markdown("### ¡Predice cómo cambia el valor de tu equipo con el tiempo!")

# Select asset class
type_selected = st.sidebar.selectbox("Selecciona el Tipo de Equipo", list(asset_options.keys()))

# Load the corresponding dataset
data_path = asset_options[type_selected]["dataset"]
df = pd.read_excel(data_path, sheet_name="Production")

# Create a simplified catalog
df_catalog = df[['Make', 'Model', 'Type']].drop_duplicates().dropna()

# Extract unique values for dropdowns and sort them
makes = sorted(df_catalog['Make'].dropna().unique())
models_by_make = {make: sorted(df_catalog[df_catalog['Make'] == make]['Model'].dropna().unique()) for make in makes}
types_by_model = {model: sorted(df_catalog[df_catalog['Model'] == model]['Type'].dropna().unique()) for model in df_catalog['Model'].unique()}

# Sidebar for user input
st.sidebar.header("🔍 Ingresar Detalles del Equipo")
selected_make = st.sidebar.selectbox("Selecciona la Marca", makes)
selected_model = st.sidebar.selectbox("Selecciona el Modelo", models_by_make[selected_make])
selected_type = st.sidebar.selectbox("Selecciona el Tipo", types_by_model[selected_model])

yearly_usage = st.sidebar.number_input("🔄 Uso Anual (Horas/Millas)", min_value=0, value=2000, step=100)
purchase_price = st.sidebar.number_input("💰 Precio de Compra Actual (MXN)", min_value=0, value=100000, step=5000)
exchange_rate = st.sidebar.number_input("💵 Tipo de Cambio", min_value=0.1, value=20.0, step=0.1)

# Load the appropriate model lazily
@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    if hasattr(pipeline.named_steps.get('model', {}), 'set_params'):
        pipeline.named_steps['model'].set_params(tree_method="hist", device="cpu")
    return pipeline

model_path = asset_options[type_selected]["model"]
pipeline = load_model(model_path)

if st.sidebar.button("📊 Predecir Precios"):
    st.subheader(f"📌 {selected_make}, {selected_model} ({selected_type})")
    
    # Generate predictions for ages 1 to 5
    results = [(0, purchase_price)]  # Add year 0 with purchase price
    for age in range(1, 6):
        usage = yearly_usage * age
        
        # Prepare input data
        input_data = pd.DataFrame({
            "Make": [selected_make],
            "Model": [selected_model],
            "Type": [selected_type],
            "Usage": [usage],
            "Unused": [0],
            "Inoperable": [0],
            "FX Rate.Fix Liq": [exchange_rate],
            "Auction Date": [pd.Timestamp.now().year],
            "Age": [age],
            "Country": ["Mexico"],
            "Auction type": ["Timed Auction"]
        })
        
        # Predict price
        predicted_price = pipeline.predict(input_data)[0]
        results.append((age, predicted_price))
    
    # Convert to DataFrame and apply Exponential Moving Average (EMA) for trendline
    results_df = pd.DataFrame(results, columns=["Age", "Predicted Price (MXN)"])
    results_df["Smoothed Price"] = results_df["Predicted Price (MXN)"].ewm(span=3, adjust=False).mean()
    results_df["% of Purchase Price"] = (results_df["Smoothed Price"] / purchase_price) * 100
    
    # Display results
    st.write("### 📈 Depreciación proyectada a 5 Años (Tendencia Suavizada)")
    df_to_display = results_df.drop(columns=["Predicted Price (MXN)"]).rename(columns={
        "Smoothed Price": "Precio Suavizado (MXN)",
        "% of Purchase Price": "% del Precio de Compra"
    }).style.format({
        "Precio Suavizado (MXN)": "{:,.2f}",
        "% del Precio de Compra": "{:.2f}%"
    })
    st.dataframe(df_to_display)
    
    # Plot results with trendline
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(results_df["Age"], results_df["Smoothed Price"], linestyle='-', marker='o', markersize=6, label="Precios Estimados", color='tab:blue')
    
    # Load watermark image
    watermark = mpimg.imread("logo.png")
    fig.figimage(watermark, xo=450, yo=400, alpha=0.5)

    ax1.set_xlabel("Edad (Años)")
    ax1.set_ylabel("Precio (MXN)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend()
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    
    for i, txt in enumerate(results_df["Smoothed Price"]):
        percent_txt = results_df["% of Purchase Price"].iloc[i]
        label_text = fr"${txt:,.2f}\n({percent_txt:.2f}%)"
        ax1.annotate(label_text, (results_df["Age"].iloc[i], results_df["Smoothed Price"].iloc[i]), textcoords="offset points", xytext=(+60,5), ha='right', fontsize=9, color='blue', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    
    st.success("✅ ¡Predicción Completa! Ajusta los valores para explorar diferentes escenarios.")
