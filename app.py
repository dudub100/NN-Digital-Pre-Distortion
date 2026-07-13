# -*- coding: utf-8 -*-
"""
Streamlit Web App: AI-Native DPD & PA Memory Analysis with QC-LDPC
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import upfirdn
import matplotlib.pyplot as plt
import math

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="AI DPD Optimizer", layout="wide")
st.title("📡 AI-Native DPD & Power Amplifier Memory Analysis")
st.markdown("Optimize Digital Pre-Distortion and evaluate residual Bit Error Rate (BER) using a Rate 0.875 QC-LDPC code.")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Simulation Parameters")
QAM_LEVEL = st.sidebar.selectbox("Modulation (QAM)", [16, 64, 256, 1024, 4096], index=2)
TARGET_OBO_DB = st.sidebar.slider("Target Output Back-Off (dB)", min_value=0.0, max_value=10.0, value=3.5, step=0.5)
OPTIMIZATION_TARGET = st.sidebar.radio("Optimization Goal", ['MSE', 'MASK'])

RRC_ALPHA = 0.12
NUM_SYMBOLS = 20000
SPS = 8
FILTER_SPAN = 20
MEMORY_DEPTH = 15

MSE_BATCH_SIZE = 256
FFT_BATCH_SIZE = 1024
active_batch_size = FFT_BATCH_SIZE if OPTIMIZATION_TARGET == 'MASK' else MSE_BATCH_SIZE
active_epochs = 100 if OPTIMIZATION_TARGET == 'MASK' else 30

# ==========================================
# CORE DSP & LDPC FUNCTIONS
# ==========================================
@st.cache_resource
def get_qc_ldpc_r0875(z=30):
    bg_p = np.array([[0, 5, 2, 10, 1, 7, 3]])
    m_bg, k_bg = bg_p.shape
    H = np.zeros((m_bg * z, (k_bg + m_bg) * z), dtype=int)
    def get_circ(shift):
        if shift < 0: return np.zeros((z, z), dtype=int)
        return np.eye(z, dtype=int)[np.roll(np.arange(z), shift)]
    for i in range(m_bg):
        for j in range(k_bg):
            H[i*z:(i+1)*z, j*z:(j+1)*z] = get_circ(bg_p[i, j])
    for i in range(m_bg):
        H[i*z:(i+1)*z, (k_bg+i)*z:(k_bg+i+1)*z] = np.eye(z)
    P = H[:, :k_bg*z]
    G = np.hstack((np.eye(k_bg*z, dtype=int), P.T))
    return H, G, k_bg*z, (k_bg+m_bg)*z

def ldpc_encode(data, G):
    return np.dot(data, G) % 2

def ldpc_decode(llrs, H, max_iter=20):
    M, N = H.shape
    V2C = np.zeros((M, N))
    check_conns = [np.where(H[i, :] == 1)[0] for i in range(M)]
    var_conns = [np.where(H[:, j] == 1)[0] for j in range(N)]
    for i in range(M): V2C[i, check_conns[i]] = llrs[check_conns[i]]
    for _ in range(max_iter):
        C2V = np.zeros((M, N))
        for i in range(M):
            idxs = check_conns[i]
            msgs = V2C[i, idxs]
            for j_idx, j in enumerate(idxs):
                others = np.delete(msgs, j_idx)
                C2V[i, j] = np.prod(np.sign(others)) * np.min(np.abs(others))
        L_total = llrs + np.sum(C2V, axis=0)
        decoded = (L_total < 0).astype(int)
        if np.all(np.dot(H, decoded) % 2 == 0): break
        for j in range(N):
            for i in var_conns[j]: V2C[i, j] = L_total[j] - C2V[i, j]
    return decoded

@st.cache_data
def get_qam_const(m_order):
    m = int(np.sqrt(m_order))
    coords = np.arange(-m+1, m, 2)
    gray = np.array([i ^ (i >> 1) for i in range(m)])
    coords = coords[np.argsort(gray)]
    grid = np.array([i + 1j*q for i in coords for q in coords])
    return grid / np.sqrt(np.mean(np.abs(grid)**2))

def calculate_llrs(rx, m_order, sigma2):
    k = int(np.log2(m_order))
    const = get_qam_const(m_order)
    llrs = []
    for s in rx:
        dist = np.abs(s - const)**2
        for b in range(k):
            mask = (np.arange(m_order) >> (k - 1 - b)) & 1
            llrs.append((np.min(dist[mask == 1]) - np.min(dist[mask == 0])) / sigma2)
    return np.array(llrs)

def get_etsi_weights(N, sps):
    freqs = np.fft.fftfreq(N)
    w = np.ones(N)
    abs_f = np.abs(freqs) * sps
    w[(abs_f > 0.4928) & (abs_f <= 0.6003)] = 5.0   
    w[(abs_f > 0.6003) & (abs_f <= 0.6765)] = 20.0  
    w[(abs_f > 0.6765) & (abs_f <= 1.5590)] = 100.0   
    w[abs_f > 1.5590] = 500.0  
    return w

def apply_rapp(v, v_sat=1.0, p=2.0):
    abs_v = np.abs(v)
    am_am = abs_v / (1.0 + (abs_v / v_sat)**(2 * p))**(1 / (2 * p))
    am_pm = 0.25 * (abs_v**2) / (1.0 + abs_v**2)
    phase_v = np.where(abs_v == 0, 0, np.angle(v))
    return am_am * np.exp(1j * (phase_v + am_pm))

def simulate_pa(x, SPS):
    eq_current = apply_rapp(x)
    x_lag1, x_lag2 = np.roll(x, SPS), np.roll(x, 2*SPS)
    x_lag1[:SPS], x_lag2[:2*SPS] = 0, 0
    sat_lag1, sat_lag2 = apply_rapp(x_lag1), apply_rapp(x_lag2)
    return eq_current + (-0.08 + 0.03j) * sat_lag1 * (np.abs(sat_lag1)**2) + (0.03 - 0.01j) * sat_lag2 * (np.abs(sat_lag2)**2)

def create_volterra_dataset(data_in, data_out, mem_depth):
    X_nn, Y_nn = [], []
    for i in range(mem_depth, len(data_in)):
        win = data_in[i-mem_depth : i+1]
        features = []
        for val in win: features.extend([val.real, val.imag, np.abs(val)**2])
        current_val = data_in[i]
        for lag in range(1, mem_depth + 1):
            past_val = data_in[i - lag]
            cross_3rd, cross_5th = current_val * (np.abs(past_val)**2), current_val * (np.abs(past_val)**4)
            features.extend([cross_3rd.real, cross_3rd.imag, cross_5th.real, cross_5th.imag])
        X_nn.append(features); Y_nn.append([data_out[i].real, data_out[i].imag])
    return np.array(X_nn), np.array(Y_nn)

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
if st.sidebar.button("🚀 Run Simulation"):
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.backend.clear_session()
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # 1. LDPC ENCODING & BASEBAND GENERATION
    status_text.text("1/6: Encoding LDPC Blocks & Generating Baseband...")
    H_ldpc, G_ldpc, K_ldpc, N_ldpc = get_qc_ldpc_r0875(z=30)
    k_qam = int(np.log2(QAM_LEVEL))
    syms_per_block = N_ldpc / k_qam
    total_blocks = math.ceil(NUM_SYMBOLS / syms_per_block)
    
    tx_data_bits = np.random.randint(0, 2, total_blocks * K_ldpc)
    tx_coded_bits = np.zeros(total_blocks * N_ldpc, dtype=int)
    for b in range(total_blocks):
        tx_coded_bits[b*N_ldpc : (b+1)*N_ldpc] = ldpc_encode(tx_data_bits[b*K_ldpc : (b+1)*K_ldpc], G_ldpc)
        
    indices = tx_coded_bits.reshape(-1, k_qam).dot(1 << np.arange(k_qam)[::-1])
    qam_syms = get_qam_const(QAM_LEVEL)[indices][:NUM_SYMBOLS]
    
    t_rrc = np.arange(-FILTER_SPAN*SPS//2, FILTER_SPAN*SPS//2 + 1) / SPS
    h_rrc = np.zeros(len(t_rrc))
    for i in range(len(t_rrc)):
        if t_rrc[i] == 0.0: h_rrc[i] = 1.0 - RRC_ALPHA + (4 * RRC_ALPHA / np.pi)
        elif RRC_ALPHA != 0 and (np.abs(t_rrc[i]) == 1.0 / (4 * RRC_ALPHA)): h_rrc[i] = (RRC_ALPHA / np.sqrt(2)) * (((1 + 2 / np.pi) * np.sin(np.pi / (4 * RRC_ALPHA))) + ((1 - 2 / np.pi) * np.cos(np.pi / (4 * RRC_ALPHA))))
        else: h_rrc[i] = (np.sin(np.pi * t_rrc[i] * (1 - RRC_ALPHA)) + 4 * RRC_ALPHA * t_rrc[i] * np.cos(np.pi * t_rrc[i] * (1 + RRC_ALPHA))) / (np.pi * t_rrc[i] * (1 - (4 * RRC_ALPHA * t_rrc[i]) ** 2))
    h_rrc /= np.sqrt(np.sum(h_rrc**2)) 
    
    raw_x = upfirdn(h_rrc, qam_syms, up=SPS)
    x_normalized = raw_x / np.max(np.abs(raw_x))
    progress_bar.progress(15)
    
    # 2. PA MODEL & OBO
    status_text.text("2/6: Simulating Hardware PA and targeting OBO...")
    sweep_in = np.linspace(0.01, 2.0, 500)
    actual_out = np.abs(simulate_pa(sweep_in + 0j, SPS=1))
    p_out_p1db_dB = 10 * np.log10(actual_out[np.argmax((10*np.log10(sweep_in**2) - 10*np.log10(actual_out**2)) >= 1.0)]**2)
    
    low, high, best_scale = 0.01, 2.0, 0.35
    for _ in range(15):
        mid = (low + high) / 2.0
        test_y = simulate_pa(x_normalized * mid, SPS)
        if (p_out_p1db_dB - 10 * np.log10(np.mean(np.abs(test_y)**2))) > TARGET_OBO_DB: low = mid
        else: high = mid
        best_scale = mid
        
    x_ideal = x_normalized * best_scale
    y_distorted = simulate_pa(x_ideal, SPS)
    progress_bar.progress(30)
    
    # 3. POLYNOMIAL DPD
    status_text.text("3/6: Extracting Memoryless Polynomial...")
    X_poly = np.column_stack([y_distorted, y_distorted * (np.abs(y_distorted)**2), y_distorted * (np.abs(y_distorted)**4), y_distorted * (np.abs(y_distorted)**6)])
    if OPTIMIZATION_TARGET == 'MSE':
        poly_coeffs, _, _, _ = np.linalg.lstsq(X_poly, x_ideal, rcond=None)
    else:
        X_poly_f = np.fft.fft(X_poly, axis=0)
        x_ideal_f = np.fft.fft(x_ideal)
        W_sqrt = np.sqrt(get_etsi_weights(len(x_ideal), SPS))
        poly_coeffs, _, _, _ = np.linalg.lstsq(X_poly_f * W_sqrt[:, None], x_ideal_f * W_sqrt, rcond=None)
    x_pred_poly = np.dot(X_poly, poly_coeffs)
    progress_bar.progress(45)
    
    # 4. NEURAL NETWORK DPD
    status_text.text(f"4/6: Training Volterra TDNN ({active_epochs} epochs)...")
    X_nn_full, Y_nn_full = create_volterra_dataset(y_distorted, x_ideal, MEMORY_DEPTH)
    trim_len = (len(X_nn_full) // active_batch_size) * active_batch_size
    X_train_full, Y_train_full = X_nn_full[:trim_len], Y_nn_full[:trim_len]
    split_idx = int(0.8 * len(X_train_full)) // active_batch_size * active_batch_size
    X_train, X_val = X_train_full[:split_idx], X_train_full[split_idx:]
    Y_train, Y_val = Y_train_full[:split_idx], Y_train_full[split_idx:]
    
    nn_model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(256, activation='tanh'), layers.Dense(128, activation='tanh'),
        layers.Dense(64, activation='tanh'), layers.Dense(2, activation='linear')
    ])
    
    if OPTIMIZATION_TARGET == 'MASK':
        w_batch_tf = tf.constant(get_etsi_weights(active_batch_size, SPS), dtype=tf.float32)
        def mask_loss_fn(y_true, y_pred):
            return tf.reduce_mean(tf.math.square(tf.math.abs(tf.signal.fft(tf.complex(y_true[:, 0] - y_pred[:, 0], y_true[:, 1] - y_pred[:, 1])))) * w_batch_tf)
        nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=mask_loss_fn)
    else:
        nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
        
    nn_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=active_epochs, batch_size=active_batch_size, shuffle=(OPTIMIZATION_TARGET=='MSE'), verbose=0)
    
    nn_pred_raw = nn_model.predict(X_nn_full, batch_size=active_batch_size, verbose=0)
    x_pred_nn = np.zeros_like(x_ideal, dtype=complex)
    x_pred_nn[MEMORY_DEPTH : MEMORY_DEPTH + len(nn_pred_raw)] = nn_pred_raw[:, 0] + 1j * nn_pred_raw[:, 1]
    progress_bar.progress(60)
    
    # 5. RECEIVER EVM EVALUATION
    status_text.text("5/6: Receiver EVM Evaluation...")
    rx_ideal = upfirdn(h_rrc, x_ideal, up=1, down=1)
    rx_distorted = upfirdn(h_rrc, y_distorted, up=1, down=1)
    rx_poly = upfirdn(h_rrc, x_pred_poly, up=1, down=1)
    rx_nn = upfirdn(h_rrc, x_pred_nn, up=1, down=1)
    
    delay = 2 * np.argmax(h_rrc)
    sym_ideal = rx_ideal[delay : delay + NUM_SYMBOLS*SPS : SPS]
    sym_distorted = rx_distorted[delay : delay + NUM_SYMBOLS*SPS : SPS]
    sym_poly = rx_poly[delay : delay + NUM_SYMBOLS*SPS : SPS]
    sym_nn = rx_nn[delay : delay + NUM_SYMBOLS*SPS : SPS]
    
    def evm(ideal, test):
        scaled = test * (np.mean(np.abs(ideal)) / np.mean(np.abs(test)))
        return 10 * np.log10(np.mean(np.abs(ideal - scaled)**2) / np.mean(np.abs(ideal)**2))
    
    evm_raw = evm(sym_ideal[MEMORY_DEPTH:], sym_distorted[MEMORY_DEPTH:])
    evm_poly = evm(sym_ideal[MEMORY_DEPTH:], sym_poly[MEMORY_DEPTH:])
    evm_nn = evm(sym_ideal[MEMORY_DEPTH:], sym_nn[MEMORY_DEPTH:])
    progress_bar.progress(75)

    # 6. LDPC DECODING & BER CALCULATION
    status_text.text("6/6: Calculating LLRs and running LDPC Min-Sum Decoder (This may take a moment)...")
    
    start_block = math.ceil(MEMORY_DEPTH / syms_per_block)
    end_block = int(len(sym_nn) // syms_per_block)
    blocks_to_test = end_block - start_block
    
    start_sym, end_sym = int(start_block * syms_per_block), int(end_block * syms_per_block)
    test_data_bits = tx_data_bits[start_block*K_ldpc : end_block*K_ldpc]

    def process_ber(ideal, test):
        const_rms = np.sqrt(np.mean(np.abs(get_qam_const(QAM_LEVEL))**2))
        t_scaled = test * (const_rms / np.sqrt(np.mean(np.abs(test)**2)))
        i_scaled = ideal * (const_rms / np.sqrt(np.mean(np.abs(ideal)**2)))
        sigma2 = np.mean(np.abs(i_scaled - t_scaled)**2)
        llrs = calculate_llrs(t_scaled, QAM_LEVEL, sigma2)
        
        errors = 0
        for b in range(blocks_to_test):
            decoded = ldpc_decode(llrs[b*N_ldpc : (b+1)*N_ldpc], H_ldpc)
            errors += np.sum(decoded[:K_ldpc] != test_data_bits[b*K_ldpc : (b+1)*K_ldpc])
        return errors / (blocks_to_test * K_ldpc)

    ber_raw = process_ber(sym_ideal[start_sym:end_sym], sym_distorted[start_sym:end_sym])
    ber_poly = process_ber(sym_ideal[start_sym:end_sym], sym_poly[start_sym:end_sym])
    ber_nn = process_ber(sym_ideal[start_sym:end_sym], sym_nn[start_sym:end_sym])
    
    # Run ILA for plotting
    X_ideal_poly = np.column_stack([x_ideal, x_ideal * (np.abs(x_ideal)**2), x_ideal * (np.abs(x_ideal)**4), x_ideal * (np.abs(x_ideal)**6)])
    pa_lin_poly = simulate_pa(np.dot(X_ideal_poly, poly_coeffs), SPS)
    
    X_ideal_nn_full, _ = create_volterra_dataset(x_ideal, x_ideal, MEMORY_DEPTH)
    nn_dpd_raw_ila = nn_model.predict(X_ideal_nn_full, batch_size=active_batch_size, verbose=0)
    x_dpd_nn_ila = np.zeros_like(x_ideal, dtype=complex)
    x_dpd_nn_ila[MEMORY_DEPTH : MEMORY_DEPTH + len(nn_dpd_raw_ila)] = nn_dpd_raw_ila[:, 0] + 1j * nn_dpd_raw_ila[:, 1]
    pa_lin_nn = simulate_pa(x_dpd_nn_ila, SPS)

    progress_bar.progress(100)
    status_text.text(f"Simulation Complete! Tested {blocks_to_test} LDPC blocks.")

    # ==========================================
    # UI METRICS AND GRAPHS
    # ==========================================
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔴 Raw PA (No DPD)")
        st.metric("EVM", f"{evm_raw:.2f} dB")
        st.metric("BER", f"{ber_raw:.2e}")
        
    with col2:
        st.subheader("🔵 Polynomial DPD")
        st.metric("EVM", f"{evm_poly:.2f} dB", f"{evm_poly - evm_raw:.2f} dB", delta_color="inverse")
        st.metric("BER", f"{ber_poly:.2e}")
        
    with col3:
        st.subheader("🟢 Volterra NN DPD")
        st.metric("EVM", f"{evm_nn:.2f} dB", f"{evm_nn - evm_raw:.2f} dB", delta_color="inverse")
        st.metric("BER", f"{ber_nn:.2e}")

    tab1, tab2 = st.tabs(["Spectrum & Constellations", "AM-AM Characteristic"])

    with tab1:
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Spectrum Plot
        psd_res = ax1.psd(x_ideal, NFFT=1024, Fs=SPS, color='black', linestyle='--', alpha=0.8, label='Ideal')
        ax1.psd(y_distorted, NFFT=1024, Fs=SPS, color='red', alpha=0.6, label='Raw PA')
        ax1.psd(pa_lin_poly, NFFT=1024, Fs=SPS, color='blue', alpha=0.6, label='Poly DPD')
        ax1.psd(pa_lin_nn, NFFT=1024, Fs=SPS, color='green', alpha=0.8, label='NN DPD')
        
        plateau_db = np.mean(10 * np.log10(psd_res[0])[np.abs(psd_res[1]) < 0.2])
        freqs_mask = np.linspace(-SPS/2, SPS/2, 1000)
        f_nodes = np.array([0, 0.440, 0.536, 0.604, 1.392, SPS/2]) * 1.12
        mask_db = np.interp(np.abs(freqs_mask), f_nodes, np.array([3.0, 3.0, -10.0, -31.0, -45.0, -45.0]) + plateau_db)
        ax1.plot(freqs_mask, mask_db, 'k-', linewidth=2, label='ETSI Mask')
        
        ax1.set_ylim(plateau_db - 60, plateau_db + 15)
        ax1.set_title('Power Spectral Density')
        ax1.legend()
        
        # Constellation Plot
        subset = 1500
        ax2.scatter(sym_distorted[MEMORY_DEPTH:subset].real, sym_distorted[MEMORY_DEPTH:subset].imag, color='r', s=2, alpha=0.4, label='Raw')
        ax2.scatter(sym_nn[MEMORY_DEPTH:subset].real, sym_nn[MEMORY_DEPTH:subset].imag, color='g', s=2, alpha=0.8, label='NN Corrected')
        ax2.set_title(f'Recovered Constellation ({QAM_LEVEL}-QAM)')
        ax2.legend()
        ax2.set_aspect('equal')
        
        st.pyplot(fig1)

    with tab2:
        fig2, ax3 = plt.subplots(figsize=(8, 6))
        ax3.scatter(np.abs(x_ideal[::10]), np.abs(y_distorted[::10]), s=1, alpha=0.3, label='Raw PA', color='red')
        ax3.scatter(np.abs(x_ideal[::10]), np.abs(pa_lin_poly[::10]), s=1, alpha=0.3, label='Poly Linearized', color='blue')
        ax3.scatter(np.abs(x_ideal[::10]), np.abs(pa_lin_nn[::10]), s=1, alpha=0.3, label='NN Linearized', color='green')
        ax3.set_title(f'AM-AM Hardware Linearization ({OPTIMIZATION_TARGET} Target)')
        ax3.set_xlabel('Input Amplitude |x|')
        ax3.set_ylabel('Output Amplitude |y|')
        ax3.legend()
        ax3.grid(True)
        st.pyplot(fig2)
