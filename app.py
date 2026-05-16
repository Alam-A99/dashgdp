# ─────────────────────────────────────────────────────────────────────
# FILE: app.py (UPDATED: No PNG Export + Table Ranking)
# DEPENDENCIES: streamlit, pandas, numpy, plotly, requests, openpyxl
# INSTALL: pip install streamlit pandas numpy plotly requests openpyxl
# RUN: streamlit run app.py
# ─────────────────────────────────────────────────────────────────────

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
@st.cache_data(ttl=3600)
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

def get_filtered_cols(df, selected_freqs, q_cols, y_cols):
    if "📊 Tahunan" in selected_freqs:
        return y_cols, "Tahunan"
    cols = []
    labels = []
    for freq in selected_freqs:
        q_num = int(freq.split()[-1])
        q_cols_filtered = sorted([c for c in q_cols if str(c).endswith(f"_{q_num}")], 
                                  key=lambda x: int(str(x).split('_')[0]))
        cols.extend(q_cols_filtered)
        labels.append(f"Q{q_num}")
    cols = sorted(cols, key=lambda x: (int(str(x).split('_')[0]), int(str(x).split('_')[1])))
    label_str = " + ".join(labels) if len(labels) > 1 else labels[0] if labels else "Kuartal"
    return cols, label_str

def calculate_growth_data(df, cols, selected_sectors):
    """Hitung growth periode terakhir vs sebelumnya untuk semua sektor terpilih"""
    growth_data = []
    for sek in selected_sectors:
        data = df.loc[df["Sektor"] == sek, cols].values.flatten()
        data = data[~np.isnan(data)]
        if len(data) < 2: continue
        g_latest = (data[-1] - data[-2]) / data[-2] * 100
        g_series = pd.Series(data).pct_change().dropna() * 100
        growth_data.append({
            "Sektor": sek, 
            "Growth_Terkini_%": g_latest, 
            "Nilai_Terkini_M": data[-1],
            "Avg_Growth_%": g_series.mean(), 
            "Volatility_σ": g_series.std(),
            "Nilai_Awal_M": data[-2]  # Untuk tabel perubahan
        })
    return pd.DataFrame(growth_data).sort_values("Growth_Terkini_%", ascending=False)

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
        help="Gunakan URL 'raw' dari GitHub"
    )
    if not raw_url or "raw" not in raw_url:
        st.warning("⚠️ Masukkan URL yang valid terlebih dahulu.")
        st.stop()

    df = load_data_from_github(raw_url)
    df, q_cols, y_cols, sektor_list = prepare_data(df)
    
    # 📅 Multi-Select Periode
    freq_options = ["Triwulan 1", "Triwulan 2", "Triwulan 3", "Triwulan 4", "📊 Tahunan"]
    selected_freqs = st.multiselect("📅 Pilih Periode", freq_options, default=["Triwulan 1"])
    if not selected_freqs:
        st.warning("⚠️ Pilih minimal satu periode.")
        st.stop()
    
    # 🏷️ Multi-Select Sektor
    selected_sectors = st.multiselect(
        "🏷️ Pilih Sektor", 
        sektor_list, 
        default=[sektor_list[0], sektor_list[1] if len(sektor_list)>1 else sektor_list[0]]
    )
    
    # 👁️ Jenis Visualisasi
    viz_type = st.selectbox("👁️ Jenis Visualisasi", [
        "📈 Tren Nilai", 
        "🏆 Perubahan Terbesar (17)", 
        "🔥 Volatilitas", 
        "📊 Komparasi Q1-Q4", 
        "🔍 Deteksi Anomali"
    ])
    
    # 🎚️ Threshold untuk deteksi anomali
    threshold = st.slider("🎯 Threshold (%)", 1.0, 20.0, 5.0)

# ─────────────────────────────────────────────────────────────────────
# 4. PROSES DATA & VISUALISASI
# ─────────────────────────────────────────────────────────────────────
cols, freq_label = get_filtered_cols(df, selected_freqs, q_cols, y_cols)
df_growth = calculate_growth_data(df, cols, selected_sectors)

tabs = st.tabs([
    "📊 Visualisasi", 
    "📋 Tabel Ranking Perubahan",  # ✅ TAB BARU: TABEL RANKING
    "📥 Export Data"
])

# ─────────────────────────────────────────────────────────────────────
# TAB 1: VISUALISASI
# ─────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader(f"📈 Analisis: {viz_type} ({freq_label})")
    
    # ─── VISUALISASI: PERUBAHAN TERBESAR (17 SEKTOR) ─────────────────
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
                template="plotly_white", height=650,
                margin=dict(l=320, r=20, t=50, b=20), bargap=0.4, showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistik format pohon (tetap dipertahankan)
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

    # ─── VISUALISASI: TREN NILAI ────────────────────────────────────
    elif viz_type == "📈 Tren Nilai":
        fig = go.Figure()
        colors = px.colors.qualitative.Set3
        for i, sek in enumerate(selected_sectors):
            data = df.loc[df["Sektor"] == sek, cols].values.flatten()
            data = data[~np.isnan(data)]
            periods = [f"{str(c).split('_')[0]} Q{str(c).split('_')[1]}" if '_' in str(c) else str(c) for c in cols]
            fig.add_trace(go.Scatter(x=periods[:len(data)], y=data, mode="lines+markers",
                                     name=sek[:40]+"...", line=dict(color=colors[i%len(colors)], width=2.5)))
        fig.update_layout(
            title=f"📈 Tren Nilai - {freq_label}", 
            xaxis_title="Periode", 
            yaxis_title="Nilai (Miliar Rupiah)",
            template="plotly_white", height=500, hovermode="x unified", xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

    # ─── VISUALISASI: VOLATILITAS ───────────────────────────────────
    elif viz_type == "🔥 Volatilitas":
        vol_data = df_growth.sort_values("Volatility_σ", ascending=False).head(15)
        fig = px.bar(vol_data, x="Volatility_σ", y="Sektor", orientation='h',
                     title=f"🔥 15 Sektor Paling Volatil - {freq_label}", 
                     color="Volatility_σ", color_continuous_scale="RdYlGn_r")
        fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    # ─── VISUALISASI: KOMPARASI Q1-Q4 ───────────────────────────────
    elif viz_type == "📊 Komparasi Q1-Q4":
        if "📊 Tahunan" in selected_freqs or len(selected_freqs) < 2:
            st.info("💡 Pilih minimal 2 Triwulan untuk komparasi.")
        else:
            latest_year = max(int(str(c).split('_')[0]) for c in cols)
            comp_data = []
            for sek in selected_sectors[:10]:
                for c in cols:
                    if str(c).startswith(f"{latest_year}_"):
                        val = df.loc[df["Sektor"] == sek, c].values[0]
                        if not pd.isna(val):
                            q = str(c).split('_')[1]
                            comp_data.append({"Sektor": sek[:35]+"..." if len(sek)>35 else sek, "Kuartal": f"Q{q}", "Nilai_M": val})
            if comp_data:
                df_comp = pd.DataFrame(comp_data)
                fig = px.bar(df_comp, x="Sektor", y="Nilai_M", color="Kuartal", barmode="group",
                             title=f"📊 Komparasi Nilai per Kuartal - Tahun {latest_year}",
                             labels={"Nilai_M": "Nilai (Miliar Rupiah)", "Sektor": "Sektor"},
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=550, xaxis_tickangle=-45, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Data tidak tersedia untuk komparasi.")

    # ─── VISUALISASI: DETEKSI ANOMALI ───────────────────────────────
    elif viz_type == "🔍 Deteksi Anomali":
        st.info(f"🔍 Threshold: {threshold}%")
        for sek in selected_sectors[:5]:
            data = df.loc[df["Sektor"] == sek, cols].values.flatten()
            data = data[~np.isnan(data)]
            if len(data) < 10: continue
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            anomalies = np.where(z_scores > 2)[0]
            status = "✅ Stabil" if len(anomalies) == 0 else f"⚠️ {len(anomalies)} anomali"
            st.markdown(f"**{sek[:50]}**: {status}")

# ─────────────────────────────────────────────────────────────────────
# TAB 2: TABEL RANKING PERUBAHAN TERBESAR & TERKECIL ✅ FITUR BARU
# ─────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader(f"📋 Tabel Ranking: Perubahan Terbesar & Terkecil ({freq_label})")
    
    if df_growth.empty:
        st.warning("⚠️ Tidak ada data pertumbuhan yang dapat ditampilkan.")
    else:
        # ─── FILTER: TAMPILKAN 17 TERBESAR + 17 TERKECIL ─────────────
        df_sorted = df_growth.sort_values("Growth_Terkini_%", ascending=False)
        top_17 = df_sorted.head(17).copy()
        bottom_17 = df_sorted.tail(17).copy()
        
        # Gabungkan: terbesar di atas, terkecil di bawah
        df_ranking = pd.concat([top_17, bottom_17]).drop_duplicates()
        
        # ─── TABEL INTERAKTIF DENGAN SORTING & SEARCH ────────────────
        st.markdown("### 🔝 17 Sektor dengan Pertumbuhan Terbesar")
        top_display = top_17[["Sektor", "Growth_Terkini_%", "Nilai_Awal_M", "Nilai_Terkini_M", "Avg_Growth_%", "Volatility_σ"]].copy()
        top_display["Growth_Terkini_%"] = top_display["Growth_Terkini_%"].apply(lambda x: f"{x:+.2f}%")
        top_display["Avg_Growth_%"] = top_display["Avg_Growth_%"].apply(lambda x: f"{x:+.2f}%")
        top_display["Nilai_Awal_M"] = top_display["Nilai_Awal_M"].apply(lambda x: f"{x:,.0f}")
        top_display["Nilai_Terkini_M"] = top_display["Nilai_Terkini_M"].apply(lambda x: f"{x:,.0f}")
        top_display["Volatility_σ"] = top_display["Volatility_σ"].apply(lambda x: f"{x:.2f}%")
        top_display.columns = ["Sektor", "Growth Terbaru", "Nilai Awal (M)", "Nilai Akhir (M)", "Rata-rata Growth", "Volatilitas (σ)"]
        st.dataframe(top_display.style.format(precision=2).background_gradient(subset=["Growth Terbaru"], cmap="Greens"), use_container_width=True, height=400)
        
        st.markdown("### 🔻 17 Sektor dengan Pertumbuhan Terkecil")
        bottom_display = bottom_17[["Sektor", "Growth_Terkini_%", "Nilai_Awal_M", "Nilai_Terkini_M", "Avg_Growth_%", "Volatility_σ"]].copy()
        bottom_display["Growth_Terkini_%"] = bottom_display["Growth_Terkini_%"].apply(lambda x: f"{x:+.2f}%")
        bottom_display["Avg_Growth_%"] = bottom_display["Avg_Growth_%"].apply(lambda x: f"{x:+.2f}%")
        bottom_display["Nilai_Awal_M"] = bottom_display["Nilai_Awal_M"].apply(lambda x: f"{x:,.0f}")
        bottom_display["Nilai_Terkini_M"] = bottom_display["Nilai_Terkini_M"].apply(lambda x: f"{x:,.0f}")
        bottom_display["Volatility_σ"] = bottom_display["Volatility_σ"].apply(lambda x: f"{x:.2f}%")
        bottom_display.columns = ["Sektor", "Growth Terbaru", "Nilai Awal (M)", "Nilai Akhir (M)", "Rata-rata Growth", "Volatilitas (σ)"]
        st.dataframe(bottom_display.style.format(precision=2).background_gradient(subset=["Growth Terbaru"], cmap="Reds"), use_container_width=True, height=400)
        
        # ─── STATISTIK FORMAT POHON (UNTUK TOP 5 & BOTTOM 5) ─────────
        st.markdown("### 📊 Ringkasan Format Pohon (Top 5 & Bottom 5)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("🔝 **Top 5 Pertumbuhan Terbesar**")
            for _, row in top_17.head(5).iterrows():
                tren = "📈 Ekspansi" if row["Growth_Terkini_%"] >= 0 else "📉 Kontraksi"
                st.text(f"🏷️  {row['Sektor'][:40]}{'...' if len(row['Sektor'])>40 else ''}")
                st.text(f"   ├─ Nilai Terkini: Rp {row['Nilai_Terkini_M']:,.0f} M")
                st.text(f"   ├─ Growth Terbaru: {row['Growth_Terkini_%']:+.2f}%")
                st.text(f"   ├─ Rata-rata Growth: {row['Avg_Growth_%']:+.2f}%")
                st.text(f"   ├─ Volatilitas (σ): {row['Volatility_σ']:.2f}%")
                st.text(f"   └─ Tren: {tren}")
                st.markdown("---")
        
        with col2:
            st.markdown("🔻 **Top 5 Pertumbuhan Terkecil**")
            for _, row in bottom_17.head(5).iterrows():
                tren = "📈 Ekspansi" if row["Growth_Terkini_%"] >= 0 else "📉 Kontraksi"
                st.text(f"🏷️  {row['Sektor'][:40]}{'...' if len(row['Sektor'])>40 else ''}")
                st.text(f"   ├─ Nilai Terkini: Rp {row['Nilai_Terkini_M']:,.0f} M")
                st.text(f"   ├─ Growth Terbaru: {row['Growth_Terkini_%']:+.2f}%")
                st.text(f"   ├─ Rata-rata Growth: {row['Avg_Growth_%']:+.2f}%")
                st.text(f"   ├─ Volatilitas (σ): {row['Volatility_σ']:.2f}%")
                st.text(f"   └─ Tren: {tren}")
                st.markdown("---")
        
        # ─── TABEL LENGKAP SEMUA SEKTOR (DENGAN SEARCH) ──────────────
        with st.expander("🔍 Lihat Tabel Lengkap Semua Sektor Terpilih"):
            all_display = df_growth[["Sektor", "Growth_Terkini_%", "Nilai_Awal_M", "Nilai_Terkini_M", "Avg_Growth_%", "Volatility_σ"]].copy()
            all_display["Growth_Terkini_%"] = all_display["Growth_Terkini_%"].apply(lambda x: f"{x:+.2f}%")
            all_display["Avg_Growth_%"] = all_display["Avg_Growth_%"].apply(lambda x: f"{x:+.2f}%")
            all_display["Nilai_Awal_M"] = all_display["Nilai_Awal_M"].apply(lambda x: f"{x:,.0f}")
            all_display["Nilai_Terkini_M"] = all_display["Nilai_Terkini_M"].apply(lambda x: f"{x:,.0f}")
            all_display["Volatility_σ"] = all_display["Volatility_σ"].apply(lambda x: f"{x:.2f}%")
            all_display.columns = ["Sektor", "Growth Terbaru", "Nilai Awal (M)", "Nilai Akhir (M)", "Rata-rata Growth", "Volatilitas (σ)"]
            st.dataframe(all_display.style.format(precision=2), use_container_width=True, height=500)

# ─────────────────────────────────────────────────────────────────────
# TAB 3: EXPORT DATA CSV
# ─────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("📥 Export Data ke CSV")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "📊 Download Data Terpilih (CSV)", 
            data=df.loc[df["Sektor"].isin(selected_sectors), ["Sektor"]+cols].to_csv(index=False).encode('utf-8'),
            file_name=f"PDB_{freq_label.replace(' + ', '_').replace(' ', '_')}_Export.csv", 
            mime="text/csv"
        )
    
    with col2:
        st.download_button(
            "📈 Download Ranking 17 (CSV)", 
            data=df_growth.head(17).to_csv(index=False).encode('utf-8'),
            file_name=f"Ranking_17_{freq_label.replace(' + ', '_').replace(' ', '_')}.csv", 
            mime="text/csv"
        )
    
    with col3:
        st.download_button(
            "📋 Download Tabel Lengkap (CSV)", 
            data=df_growth.to_csv(index=False).encode('utf-8'),
            file_name=f"Semua_Sektor_{freq_label.replace(' + ', '_').replace(' ', '_')}.csv", 
            mime="text/csv"
        )
    
    st.info("💡 File CSV dapat dibuka di Excel, Google Sheets, atau aplikasi spreadsheet lainnya.")

# Footer dengan heart symbol ❤️
st.caption("💡 Keep on Learning in deep heart with ❤️.")
