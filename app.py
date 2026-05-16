# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from io import BytesIO
import re

st.set_page_config(page_title="📊 Dashboard PDB", layout="wide", page_icon="📈")

# ─────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)  #
def load_data_from_github(raw_url):
    try:
        res = requests.get(raw_url, timeout=10)
        res.raise_for_status()
        df = pd.read_excel(BytesIO(res.content), header=0)
        df.rename(columns={df.columns[0]: "Sektor"}, inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────────────
# 2. FUNGSI PREPARASI DATA
# ─────────────────────────────────────────────────────────────────────
def prepare_data(df):
    all_cols = [c for c in df.columns[1:] if isinstance(c, (str, int))]
    q_cols = sorted([c for c in all_cols if re.match(r"^\d{4}_[1-4]$", str(c))], 
                    key=lambda x: (int(str(x).split('_')[0]), int(str(x).split('_')[1])))
    y_cols = sorted([c for c in all_cols if re.match(r"^\d{4}$", str(c))], key=lambda x: int(str(x)))
    sektor_list = df["Sektor"].dropna().tolist()
    return df, q_cols, y_cols, sektor_list

def get_filtered_cols(df, freq, q_cols, y_cols):
    if freq == "📊 Tahunan":
        return y_cols, "Tahunan"
    q_num = int(freq.split()[-1])
    cols = sorted([c for c in q_cols if str(c).endswith(f"_{q_num}")], key=lambda x: int(str(x).split('_')[0]))
    return cols, f"Triwulan {q_num}"

def calculate_growth_data(df, cols, selected_sectors):
    growth_data = []
    for sek in selected_sectors:
        data = df.loc[df["Sektor"] == sek, cols].values.flatten()
        data = data[~np.isnan(data)]
        if len(data) < 2: continue
        g_latest = (data[-1] - data[-2]) / data[-2] * 100
        g_series = pd.Series(data).pct_change().dropna() * 100
        growth_data.append({
            "Sektor": sek, "Growth_Terkini_%": g_latest, "Nilai_Terkini_M": data[-1],
            "Avg_Growth_%": g_series.mean(), "Volatility_σ": g_series.std()
        })
    return pd.DataFrame(growth_data).sort_values("Growth_Terkini_%", ascending=False)

def detect_anomalies(df, cols, selected_sectors, method, threshold):
    results = []
    for sek in selected_sectors:
        data = df.loc[df["Sektor"] == sek, cols].values.flatten()
        data = data[~np.isnan(data)]
        if len(data) < 10: continue
        
        anomalies = []
        if method == "Z-Score (±2σ)":
            z = np.abs((data - np.mean(data)) / np.std(data))
            anomalies = data.index[z > threshold].tolist()
        elif method == "IQR Method":
            Q1, Q3 = np.percentile(data, 25), np.percentile(data, 75)
            IQR = Q3 - Q1
            lower, upper = Q1 - threshold * IQR, Q3 + threshold * IQR
            anomalies = data.index[(data < lower) | (data > upper)].tolist()
        elif method == "Growth Spike":
            g = np.diff(data) / data[:-1] * 100
            anomalies = np.where(np.abs(g) > threshold)[0] + 1
            
        results.append({"Sektor": sek, "Anomalies": len(anomalies), "Data": data})
    return results

# ─────────────────────────────────────────────────────────────────────
# 3. SIDEBAR & KONTROL
# ─────────────────────────────────────────────────────────────────────
st.title("📊 Dashboard Analisis PDB")
st.caption("Eksplorasi real-time dataset PDB menurut Lapangan Usaha (ADHK)")

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    # 🔗 INPUT GITHUB RAW URL
    raw_url = st.text_input(
        "🔗 DATA URL",
        value="https://raw.githubusercontent.com/username/repo/main/PDB_Seri2010.xlsx",
        help="Gunakan URL 'raw' dari GitHub. 'raw'"
    )
    if not raw_url or "raw" not in raw_url:
        st.warning("⚠️ Masukkan URL yang valid terlebih dahulu.")
        st.stop()

    df = load_data_from_github(raw_url)
    df, q_cols, y_cols, sektor_list = prepare_data(df)
    
    freq = st.selectbox("📅 Periode", ["Triwulan 1", "Triwulan 2", "Triwulan 3", "Triwulan 4", "📊 Tahunan"])
    selected_sectors = st.multiselect("🏷️ Pilih Sektor", sektor_list, default=[sektor_list[0], sektor_list[1] if len(sektor_list)>1 else sektor_list[0]])
    
    viz_type = st.selectbox("👁️ Jenis Visualisasi", [
        "📈 Tren Nilai", "🏆 Perubahan Terbesar (17)", "🔥 Volatilitas", 
        "📊 Komparasi Q1-Q4", "🔍 Deteksi Anomali"
    ])
    threshold = st.slider("🎯 Threshold (%)", 1.0, 20.0, 5.0)

# ─────────────────────────────────────────────────────────────────────
# 4. PROSES DATA & VISUALISASI
# ─────────────────────────────────────────────────────────────────────
cols, freq_label = get_filtered_cols(df, freq, q_cols, y_cols)
df_growth = calculate_growth_data(df, cols, selected_sectors)

tabs = st.tabs(["📊 Visualisasi", "📋 Tabel Perubahan", "📥 Export"])

with tabs[0]:
    st.subheader(f"📈 Analisis: {viz_type} ({freq_label})")
    
    if viz_type == "🏆 Perubahan Terbesar (17)":
        df_top17 = df_growth.head(17)
        if df_top17.empty:
            st.warning("⚠️ Tidak cukup data untuk 17 sektor.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_top17["Sektor"].apply(lambda x: x[:45]+"..." if len(x)>45 else x),
                x=df_top17["Growth_Terkini_%"], orientation='h',
                marker_color=df_top17["Growth_Terkini_%"].apply(lambda v: "#2A9D8F" if v >= 0 else "#E63946"),
                marker_line=dict(width=1, color='white'),
                hovertemplate="<b>%{y}</b><br>Perubahan: %{x:.2f}%<br>Nilai: Rp %{customdata:,.0f} M<extra></extra>",
                customdata=df_top17["Nilai_Terkini_M"]
            ))
            fig.update_layout(
                title=f"🏆 Ranking 17 Sektor: Perubahan Terbesar - {freq_label}",
                xaxis_title="Pertumbuhan Periode Terakhir (%)",
                yaxis=dict(autorange="reversed", title="Sektor"),
                template="plotly_white", height=650, margin=dict(l=320, r=20, t=50, b=20), bargap=0.4, showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistik format pohon
            st.markdown("### 📊 Ringkasan Statistik")
            for _, row in df_top17.iterrows():
                tren = "📈 Ekspansi" if row["Growth_Terkini_%"] >= 0 else "📉 Kontraksi"
                st.text(f"🏷️  {row['Sektor'][:50]}{'...' if len(row['Sektor'])>50 else ''}")
                st.text(f"   ├─ Nilai Terkini: Rp {row['Nilai_Terkini_M']:,.0f} M")
                st.text(f"   ├─ Growth Terbaru: {row['Growth_Terkini_%']:+.2f}%")
                st.text(f"   ├─ Rata-rata Growth: {row['Avg_Growth_%']:+.2f}%")
                st.text(f"   ├─ Volatilitas (σ): {row['Volatility_σ']:.2f}%")
                st.text(f"   └─ Tren: {tren}")
                st.markdown("---")

    elif viz_type == "📈 Tren Nilai":
        fig = go.Figure()
        colors = px.colors.qualitative.Set3
        for i, sek in enumerate(selected_sectors):
            data = df.loc[df["Sektor"] == sek, cols].values.flatten()
            data = data[~np.isnan(data)]
            periods = [str(c).replace('_', ' Q') if '_' in str(c) else str(c) for c in cols]
            fig.add_trace(go.Scatter(x=periods[:len(data)], y=data, mode="lines+markers",
                                     name=sek[:40]+"...", line=dict(color=colors[i%len(colors)], width=2.5)))
        fig.update_layout(title=f"📈 Tren Nilai - {freq_label}", xaxis_title="Periode", yaxis_title="Nilai (Miliar Rupiah)",
                          template="plotly_white", height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "🔥 Volatilitas":
        vol_data = df_growth.sort_values("Volatility_σ", ascending=False).head(15)
        fig = px.bar(vol_data, x="Volatility_σ", y="Sektor", orientation='h',
                     title=f"🔥 15 Sektor Paling Volatil - {freq_label}", color="Volatility_σ", color_continuous_scale="RdYlGn_r")
        fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "🔍 Deteksi Anomali":
        anomaly_res = detect_anomalies(df, cols, selected_sectors, "IQR Method" if threshold > 2 else "Z-Score (±2σ)", threshold)
        st.info(f"🔍 Metode: IQR/Z-Score | Threshold: {threshold}")
        for res in anomaly_res:
            if res["Anomalies"] > 0:
                st.warning(f"⚠️ **{res['Sektor'][:50]}**: Terdeteksi `{res['Anomalies']}` periode anomali")
            else:
                st.success(f"✅ **{res['Sektor'][:50]}**: Data stabil")

with tabs[1]:
    st.subheader("📋 Tabel Perubahan Kuartalan/Tahunan")
    # Buat tabel perubahan periode-to-periode
    change_rows = []
    for sek in selected_sectors:
        data = df.loc[df["Sektor"] == sek, cols].values.flatten()
        data = data[~np.isnan(data)]
        periods = [str(c) for c in cols]
        for i in range(1, len(data)):
            change_rows.append({
                "Sektor": sek, "Periode_Dari": periods[i-1], "Periode_Ke": periods[i],
                "Nilai_Awal_M": data[i-1], "Nilai_Akhir_M": data[i],
                "Perubahan_%": ((data[i]-data[i-1])/data[i-1]*100)
            })
    df_changes = pd.DataFrame(change_rows)
    if not df_changes.empty:
        df_display = df_changes[["Sektor", "Periode_Dari", "Periode_Ke", "Nilai_Awal_M", "Nilai_Akhir_M", "Perubahan_%"]].copy()
        df_display["Perubahan_%"] = df_display["Perubahan_%"].apply(lambda x: f"{x:+.2f}%")
        st.dataframe(df_display, use_container_width=True, height=500)
    else:
        st.info("Pilih minimal 1 sektor dengan data lengkap.")

with tabs[2]:
    st.subheader("📥 Export Data")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📊 Download Data Terpilih (CSV)", 
                           data=df.loc[df["Sektor"].isin(selected_sectors), ["Sektor"]+cols].to_csv(index=False).encode('utf-8'),
                           file_name=f"PDB_{freq_label.replace(' ', '_')}_Export.csv", mime="text/csv")
    with col2:
        st.download_button("📈 Download Ranking 17 (CSV)", 
                           data=df_growth.head(17).to_csv(index=False).encode('utf-8'),
                           file_name=f"Ranking_17_{freq_label.replace(' ', '_')}.csv", mime="text/csv")

st.caption("💡 Keep on Learning in deep heart with Love.")
