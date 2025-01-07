import streamlit as st
import pandas as pd
import networkx as nx
from streamlit_folium import folium_static
import folium
import numpy as np
from folium.plugins import AntPath
from datetime import datetime, timedelta

# Pengaturan halaman
st.set_page_config(
    page_title="Dashboard Bus Surabaya",
    page_icon="🚌",
    layout="wide"
)

# Judul dan deskripsi
st.title("🚌 Dashboard Bus Surabaya")
st.text("Monitoring dan Analisis Layanan Bus Kota Surabaya")

# Data halte dari CSV
data_path = "rute bis surabaya_2.csv"
data_dtf = pd.read_csv(data_path)

halte_dtf = data_dtf[["Halte_asal", "Latitude", "Longitude"]].drop_duplicates()
edges_dtf = data_dtf[["Antar_halte", "Halte_tujuan", "waktu_antar_halte"]]

# Periksa apakah ada edge yang menghubungkan node asal dan tujuan
valid_edges = edges_dtf.dropna(subset=["Antar_halte", "Halte_tujuan", "waktu_antar_halte"])
if valid_edges.empty:
    raise ValueError("Tidak ada edge valid dalam data untuk membentuk graf.")

# Bersihkan nama halte dari spasi tambahan
halte_dtf["Halte_asal"] = halte_dtf["Halte_asal"].str.strip()
edges_dtf["Antar_halte"] = edges_dtf["Antar_halte"].str.strip()
edges_dtf["Halte_tujuan"] = edges_dtf["Halte_tujuan"].str.strip()

# Pastikan semua halte memiliki koordinat
valid_halte = halte_dtf.dropna(subset=["Latitude", "Longitude"])  # Ambil halte dengan koordinat valid

# Buat posisi node dengan nama halte yang sudah bersih
pos = {node: (row["Longitude"], row["Latitude"]) for node, row in valid_halte.set_index("Halte_asal").iterrows()}

# Sidebar untuk filter
st.sidebar.header("Filter Data")

# Pilihan halte keberangkatan dan kedatangan
halte_list = halte_dtf["Halte_asal"].unique()
halte_keberangkatan = st.sidebar.selectbox("Halte Keberangkatan", halte_list)
halte_kedatangan = st.sidebar.selectbox("Halte Kedatangan", halte_list)

# Filter tanggal
selected_date = st.sidebar.date_input("Tanggal Hari Ini", datetime.now())
selected_time = st.sidebar.time_input("Pilih Waktu", datetime.now().time())

# Tampilkan informasi rute yang dipilih
st.sidebar.markdown("---")
st.sidebar.subheader("Informasi Rute")
st.sidebar.write(f"Dari: {halte_keberangkatan}")
st.sidebar.write(f"Ke: {halte_kedatangan}")

# Layout dengan kolom
col1, col2, col3 = st.columns(3)

# Metrik utama
with col1:
    st.metric(label="Total Penumpang Hari Ini", value="1,234", delta="12%")
with col2:
    st.metric(label="Bus Beroperasi", value="45", delta="-2")
with col3:
    st.metric(label="Rata-rata Waktu Tunggu", value="8 menit", delta="-1 menit")

# Bangun graf menggunakan NetworkX
G = nx.DiGraph()  # Graf berarah
for _, row in edges_dtf.iterrows():
    G.add_edge(row["Antar_halte"], row["Halte_tujuan"], weight=row["waktu_antar_halte"])

# Filter graf untuk hanya memasukkan node dengan posisi valid
nodes_with_pos = set(pos.keys())
G_filtered = G.subgraph(nodes_with_pos).copy()

def calculate_route_time(route, G):
    total_time = 0
    for i in range(len(route) - 1):
        try:
            total_time += G[route[i]][route[i + 1]]['weight']
        except KeyError:
            raise ValueError(f"Tidak ada edge antara {route[i]} dan {route[i + 1]} di graf.")
    return total_time

# Fungsi Algoritma Genetika untuk menemukan rute optimal
def bfs_shortest_path(G, start, end):
    if start not in G.nodes() or end not in G.nodes():
        raise ValueError("Halte keberangkatan atau tujuan tidak valid")

    queue = [(start, [start])]
    while queue:
        current_node, path = queue.pop(0)
        if current_node == end:
            total_time = calculate_route_time(path, G)
            return path, total_time
        for neighbor in G.neighbors(current_node):
            if neighbor not in path:
                queue.append((neighbor, path + [neighbor]))
    return [], float('inf')

# Fungsi untuk membaca data rute dan lalu lintas
route_data = pd.read_csv("rute bis surabaya.csv")
traffic_data = pd.read_csv("Traffic.csv")

def preprocess_traffic_data(traffic_data):
    # Gabungkan kolom Date dan Time menjadi satu kolom datetime
    traffic_data["Datetime"] = pd.to_datetime(
        traffic_data["Date"].astype(str) + " " + traffic_data["Time"],
        format="%d %I:%M:%S %p",  # Format sesuai contoh: "10 12:15:00 AM"
        errors="coerce"  # Menangani error parsing
    ) 
    # Ekstrak jam dan menit dari kolom Datetime
    traffic_data["Hour"] = traffic_data["Datetime"].dt.hour
    traffic_data["Minute"] = traffic_data["Datetime"].dt.minute

    situation_mapping = {"low": 0, "normal": 1, "high": 2, "heavy": 3}
    traffic_data["Traffic_Situation_Code"] = traffic_data["Traffic Situation"].map(situation_mapping)

    return traffic_data

traffic_datapre = preprocess_traffic_data(traffic_data)

# Feature columns and target column
feature_columns = ["Hour", "Minute"]
target_column = "Traffic_Situation_Code"

# Convert data to numpy arrays
X = traffic_data[feature_columns].values
y = traffic_data[target_column].values

# KNN implementation
def knn_predict(X_train, y_train, X_test, k=3):
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    predictions = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        predictions.append(most_common)
    return predictions

# Traffic situation decoding
situation_decoding = {0: "low", 1: "normal", 2: "high", 3: "heavy"}

# Predict traffic situation based on departure time
def predict_traffic_situation(departure_time, traffic_data):
    # Preprocess traffic data
    processed_traffic_data = preprocess_traffic_data(traffic_data)
    
    # Extract features and target
    X = processed_traffic_data[["Hour", "Minute"]].values
    y = processed_traffic_data["Traffic_Situation_Code"].values
    
    # Prepare test data
    hour = departure_time.hour
    minute = departure_time.minute
    X_test = np.array([[hour, minute]])

    # Predict using KNN
    prediction_code = knn_predict(X, y, X_test, k=3)[0]
    return situation_decoding[prediction_code]

# Integrate with bus schedule
def calculate_dynamic_bus_schedule(route_data, traffic_data, start_stop, end_stop, selected_date, selected_time, optimal_route):
    """
    Calculate bus schedule based on the optimal route found by BFS
    """
    start_time = datetime.combine(selected_date, selected_time)
    
    # Convert waktu_antar_halte to numeric
    route_data["waktu_antar_halte"] = pd.to_numeric(route_data["waktu_antar_halte"], errors="coerce")
    
    # Create a dictionary for route times between stops
    route_time = {
        (row["Halte_asal"], row["Halte_tujuan"]): row["waktu_antar_halte"]
        for _, row in route_data.iterrows()
    }
    
    schedule = []
    current_time = start_time
    
    # Use the optimal route to calculate schedule
    for i in range(len(optimal_route) - 1):
        current_stop = optimal_route[i]
        next_stop = optimal_route[i + 1]
        
        # Get base travel time between stops
        travel_time = route_time.get((current_stop, next_stop), 0)
        
        # Predict traffic situation and adjust travel time
        traffic_status = predict_traffic_situation(current_time, traffic_data)
        
        # Apply traffic multiplier
        if traffic_status == "low":
            travel_time *= 0.8
            traffic_description = "lebih cepat"
        elif traffic_status == "normal":
            traffic_description = "tepat waktu"
        elif traffic_status == "high":
            travel_time *= 1.3
            traffic_description = "sedikit lebih lama"
        elif traffic_status == "heavy":
            travel_time *= 1.5
            traffic_description = "telat"
        
        # Round travel time to 2 decimal places
        travel_time = round(travel_time, 2)
        
        # Calculate arrival time
        arrival_time = current_time + timedelta(minutes=travel_time)
        
        # Add segment to schedule
        schedule.append({
            "Waktu_Keberangkatan": current_time.strftime("%I:%M:%S %p"),
            "Halte_Keberangkatan": current_stop,
            "Halte_Tujuan": next_stop,
            "Waktu_Kedatangan": arrival_time.strftime("%I:%M:%S %p"),
            "Status_Kedatangan": traffic_description
        })
        
        # Update current time for next segment
        current_time = arrival_time
    
    return pd.DataFrame(schedule)

# Example usage
try:
    # Get optimal route first
    best_route, best_time = bfs_shortest_path(G_filtered, halte_keberangkatan, halte_kedatangan)
    
    # Calculate schedule based on optimal route
    bus_schedule = calculate_dynamic_bus_schedule(
        route_data, 
        traffic_data, 
        halte_keberangkatan, 
        halte_kedatangan, 
        selected_date, 
        selected_time,
        best_route
    )
    
    # Display schedule
    st.subheader("Jadwal Bus - Rute Optimal")
    st.dataframe(bus_schedule)
    
    # Calculate and display total travel time
    if not bus_schedule.empty:
        first_departure = datetime.strptime(bus_schedule.iloc[0]["Waktu_Keberangkatan"], "%I:%M:%S %p")
        last_arrival = datetime.strptime(bus_schedule.iloc[-1]["Waktu_Kedatangan"], "%I:%M:%S %p")
        total_time = (last_arrival - first_departure).total_seconds() / 60
        
        st.write("### Ringkasan Perjalanan")
        st.write(f"Rute: {' → '.join(best_route)}")
        st.write(f"Total Waktu Perjalanan: {round(total_time, 2)} menit")
        st.write(f"Waktu Keberangkatan: {bus_schedule.iloc[0]['Waktu_Keberangkatan']}")
        st.write(f"Waktu Kedatangan: {bus_schedule.iloc[-1]['Waktu_Kedatangan']}")
    
    # Display map
    center_lat = valid_halte["Latitude"].mean()
    center_lon = valid_halte["Longitude"].mean()
    folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Add markers for bus stops
    for _, row in halte_dtf.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=row["Halte_asal"],
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(folium_map)

    # Add route animation
    route_coordinates = [(pos[node][1], pos[node][0]) for node in best_route]
    AntPath(locations=route_coordinates, color="green", weight=5, delay=300).add_to(folium_map)

    # Display map
    st.subheader("Visualisasi Rute")
    folium_static(folium_map)

except Exception as e:
    st.error(f"Gagal menghitung jadwal: {str(e)}")






