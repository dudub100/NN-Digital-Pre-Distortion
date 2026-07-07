# -*- coding: utf-8 -*-
"""
Streamlit Web App: AI-Native DPD & PA Memory Analysis
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import upfirdn
import matplotlib.pyplot as plt

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="AI DPD Optimizer", layout="wide")
st.title("📡 AI-Native DPD & Power Amplifier Memory Analysis")
st.markdown("Optimize Digital Pre-Distortion using Time-Domain MSE or Frequency-Domain ETSI Mask constraints.")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Simulation Parameters")
QAM_LEVEL = st.sidebar.selectbox("Modulation (QAM)", [16, 64, 256, 1024, 4096], index=1)
TARGET_OBO_DB = st.sidebar.slider("Target Output Back-Off (dB)", min_value=0.0, max_value=10.0, value=4.0, step=0.5)
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
# CORE DSP FUNCTIONS
# ==========================================
def get_etsi_weights(N, sps):
    freqs = np.fft.fftfreq(N)
    w = np.ones(N)
    rs_norm = 1.0 / sps
    CS_margin = 1.12 
    f1, f2, f3, f4 = 0.440 * CS_margin, 0.536 * CS_margin, 0.604 * CS_margin, 1.392 * CS_margin
    abs_f = np.abs(freqs) / rs_norm
    
    w[(abs_f > f1) & (abs_f <= f2)] = 5.0   
    w[(abs_f > f2) & (abs_f <= f3)] = 20.0  
    w[(abs_f > f3) & (abs_f <= f4)] = 100.0   
    w[abs_f > f4] = 500.0  
    return w

def apply_rapp(v, v_sat=1.0, p=2.0):
    abs_v = np.abs(v)
    am_am = abs_v / (1.0 + (abs_v / v_sat)**(2 * p))**(1 / (2 * p))
    am_pm = 0.25 * (abs_v**2) / (1.0 + abs_v**2)
    phase_v = np.where(abs_v == 0, 0, np.angle(v))
    return am_am * np.exp(1j * (phase_v + am_pm))

def simulate_pa(x, SPS):
    eq_current = apply_rapp(x)
    m1_lag1, m1_lag2 = -0.08 + 0.03j, 0.03 - 0.01j
    x_lag1, x_lag2 = np.roll(x, SPS), np.roll(x, 2*SPS)
    x_lag1[:SPS], x_lag2[:2*SPS] = 0, 0
    sat_lag1, sat_lag2 = apply_rapp(x_lag1), apply_rapp(x_lag2)
    eq_memory = m1_lag1 * sat_lag1 * (np.abs(sat_lag1)**2) + m1_lag2 * sat_lag2 * (np.abs(sat_lag2)**2)
    return eq_current + eq_memory

def create_volterra_dataset(data_in, data_out, mem_depth):
    N = len(data_in)
    X_nn, Y_nn = [], []
    for i in range(mem_depth, N):
        window = data_in[i-mem_depth : i+1]
        features = []
        for val in window:
            features.extend([val.real, val.imag, np.abs(val)**2])
        current_val = data_in[i]
        for lag in range(1, mem_depth + 1):
            past_val = data_in[i - lag]
            cross_3rd = current_val * (np.abs(past_val)**2)
            cross_5th = current_val * (np.abs(past_val)**4)
            features.extend([cross_3rd.real, cross_3rd.imag, cross_5th.real, cross_5th.imag])
        X_nn.append(features)
        Y_nn.append([data_out[i].real, data_out[i].imag])
    return np.array(X_nn), np.array(Y_nn)

# ==========================================
# TWO TONE ANALYSIS (IM3 & MEMORY ASYMMETRY)
# ==========================================
def analyze_im3_and_memory():
    N_tt = 8192
    t = np.arange(N_tt)
    window = np.blackman(N_tt)
    
    # 1. Power Sweep (AM-AM & IM3 Asymptotes)
    pin_db = np.linspace(-20, 5, 20)
    pout_fund, pout_im3 = [], []
    
    # Snap frequencies exactly to FFT bins to completely eliminate spectral leakage
    k1, k2 = int(0.04 * N_tt), int(0.05 * N_tt)
    f1, f2 = k1 / N_tt, k2 / N_tt
    
    for p in pin_db:
        amp = 10**(p/20.0)
        x = amp * (np.exp(1j * 2 * np.pi * f1 * t) + np.exp(1j * 2 * np.pi * f2 * t))
        y = simulate_pa(x, SPS)
        
        # Apply window to suppress start-of-burst transient leakage
        Y_f = np.fft.fft(y * window) / np.sum(window)
        freqs = np.fft.fftfreq(N_tt)
        
        idx_f1, idx_f2 = np.argmin(np.abs(freqs - f1)), np.argmin(np.abs(freqs - f2))
        idx_im3_l, idx_im3_u = np.argmin(np.abs(freqs - (2*f1 - f2))), np.argmin(np.abs(freqs - (2*f2 - f1)))
        
        # Lower noise floor to -200 dB (1e-20) to reveal true 3:1 small-signal slope
        p_fund = 10*np.log10((np.abs(Y_f[idx_f1])**2 + np.abs(Y_f[idx_f2])**2)/2 + 1e-20)
        p_im3 = 10*np.log10((np.abs(Y_f[idx_im3_l])**2 + np.abs(Y_f[idx_im3_u])**2)/2 + 1e-20)
        
        pout_fund.append(p_fund)
        pout_im3.append(p_im3)
        
    # 2. Frequency Sweep (Memory Asymmetry)
    df_sweep = np.linspace(0.005, 0.1, 30)
    im3_l_arr, im3_u_arr = [], []
    amp = 10**(-2/20.0) # Near saturation
    
    for df in df_sweep:
        f_center = 0.045
        k1_s = int((f_center - df/2) * N_tt)
        k2_s = int((f_center + df/2) * N_tt)
        f1_s, f2_s = k1_s / N_tt, k2_s / N_tt
        
        x = amp * (np.exp(1j * 2 * np.pi * f1_s * t) + np.exp(1j * 2 * np.pi * f2_s * t))
        y = simulate_pa(x, SPS)
        
        Y_f = np.fft.fft(y * window) / np.sum(window)
        freqs = np.fft.fftfreq(N_tt)
        
        idx_im3_l, idx_im3_u = np.argmin(np.abs(freqs - (2*f1_s - f2_s))), np.argmin(np.abs(freqs - (2*f2_s - f1_s)))
        
        im3_l_arr.append(10*np.log10(np.abs(Y_f[idx_im3_l])**2 + 1e-20))
        im3_u_arr.append(10*np.log10(np.abs(Y_f[idx_im3_u])**2 + 1e-20))

    return pin_db, pout_fund, pout_im3, df_sweep, im3_l_arr, im3_u_arr

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
if st.sidebar.button("🚀 Run Simulation"):
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.backend.clear_session()
    
    # UI Progress
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # 1. GENERATE SIGNAL
    status_text.text("1/5: Generating 64-QAM Baseband...")
    m_pam = int(np.sqrt(QAM_LEVEL))
    pam_levels = np.arange(-m_pam + 1, m_pam + 1, 2)
    qam_syms = np.random.choice(pam_levels, NUM_SYMBOLS) + 1j * np.random.choice(pam_levels, NUM_SYMBOLS)
    
    t_rrc = np.arange(-FILTER_SPAN*SPS//2, FILTER_SPAN*SPS//2 + 1) / SPS
    h_rrc = np.zeros(len(t_rrc))
    for i in range(len(t_rrc)):
        if t_rrc[i] == 0.0: h_rrc[i] = 1.0 - RRC_ALPHA + (4 * RRC_ALPHA / np.pi)
        elif RRC_ALPHA != 0 and (np.abs(t_rrc[i]) == 1.0 / (4 * RRC_ALPHA)): h_rrc[i] = (RRC_ALPHA / np.sqrt(2)) * (((1 + 2 / np.pi) * np.sin(np.pi / (4 * RRC_ALPHA))) + ((1 - 2 / np.pi) * np.cos(np.pi / (4 * RRC_ALPHA))))
        else: h_rrc[i] = (np.sin(np.pi * t_rrc[i] * (1 - RRC_ALPHA)) + 4 * RRC_ALPHA * t_rrc[i] * np.cos(np.pi * t_rrc[i] * (1 + RRC_ALPHA))) / (np.pi * t_rrc[i] * (1 - (4 * RRC_ALPHA * t_rrc[i]) ** 2))
    h_rrc /= np.sqrt(np.sum(h_rrc**2)) 
    
    raw_x = upfirdn(h_rrc, qam_syms, up=SPS)
    x_normalized = raw_x / np.max(np.abs(raw_x))
    progress_bar.progress(20)
    
    # 2. PA MODEL & OBO
    status_text.text("2/5: Simulating Hardware PA and targeting OBO...")
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
    progress_bar.progress(40)
    
    # 3. POLYNOMIAL DPD
    status_text.text("3/5: Extracting Memoryless Polynomial...")
    X_poly = np.column_stack([y_distorted, y_distorted * (np.abs(y_distorted)**2), y_distorted * (np.abs(y_distorted)**4), y_distorted * (np.abs(y_distorted)**6)])
    if OPTIMIZATION_TARGET == 'MSE':
        poly_coeffs, _, _, _ = np.linalg.lstsq(X_poly, x_ideal, rcond=None)
    else:
        X_poly_f = np.fft.fft(X_poly, axis=0)
        x_ideal_f = np.fft.fft(x_ideal)
        W_sqrt = np.sqrt(get_etsi_weights(len(x_ideal), SPS))
        poly_coeffs, _, _, _ = np.linalg.lstsq(X_poly_f * W_sqrt[:, None], x_ideal_f * W_sqrt, rcond=None)
    x_pred_poly = np.dot(X_poly, poly_coeffs)
    progress_bar.progress(60)
    
    # 4. NEURAL NETWORK DPD
    status_text.text(f"4/5: Training Volterra TDNN ({active_epochs} epochs)...")
    X_nn_full, Y_nn_full = create_volterra_dataset(y_distorted, x_ideal, MEMORY_DEPTH)
    trim_len = (len(X_nn_full) // active_batch_size) * active_batch_size
    X_train_full, Y_train_full = X_nn_full[:trim_len], Y_nn_full[:trim_len]
    split_idx = int(0.8 * len(X_train_full)) // active_batch_size * active_batch_size
    X_train, X_val = X_train_full[:split_idx], X_train_full[split_idx:]
    Y_train, Y_val = Y_train_full[:split_idx], Y_train_full[split_idx:]
    
    nn_model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(256, activation='tanh'),
        layers.Dense(128, activation='tanh'),
        layers.Dense(64, activation='tanh'),
        layers.Dense(2, activation='linear')
    ])
    
    if OPTIMIZATION_TARGET == 'MASK':
        w_batch_tf = tf.constant(get_etsi_weights(active_batch_size, SPS), dtype=tf.float32)
        def mask_loss_fn(y_true, y_pred):
            err_c = tf.complex(y_true[:, 0] - y_pred[:, 0], y_true[:, 1] - y_pred[:, 1])
            return tf.reduce_mean(tf.math.square(tf.math.abs(tf.signal.fft(err_c))) * w_batch_tf)
        nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=mask_loss_fn)
    else:
        nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
        
    nn_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=active_epochs, batch_size=active_batch_size, shuffle=(OPTIMIZATION_TARGET=='MSE'), verbose=0)
    
    nn_pred_raw = nn_model.predict(X_nn_full, batch_size=active_batch_size, verbose=0)
    x_pred_nn = np.zeros_like(x_ideal, dtype=complex)
    x_pred_nn[MEMORY_DEPTH : MEMORY_DEPTH + len(nn_pred_raw)] = nn_pred_raw[:, 0] + 1j * nn_pred_raw[:, 1]
    progress_bar.progress(80)
    
    # 5. RECEIVER EVALUATION & IM3 ANALYSIS
    status_text.text("5/5: Receiver Evaluation and Two-Tone Tests...")
    
    # Calculate EVM
    rx_ideal = upfirdn(h_rrc, x_ideal, up=1, down=1)
    rx_distorted = upfirdn(h_rrc, y_distorted, up=1, down=1)
    rx_poly = upfirdn(h_rrc, x_pred_poly, up=1, down=1)
    rx_nn = upfirdn(h_rrc, x_pred_nn, up=1, down=1)
    
    delay = 2 * np.argmax(h_rrc)
    sym_ideal = rx_ideal[delay : delay + NUM_SYMBOLS*SPS : SPS][MEMORY_DEPTH:]
    sym_distorted = rx_distorted[delay : delay + NUM_SYMBOLS*SPS : SPS][MEMORY_DEPTH:]
    sym_poly = rx_poly[delay : delay + NUM_SYMBOLS*SPS : SPS][MEMORY_DEPTH:]
    sym_nn = rx_nn[delay : delay + NUM_SYMBOLS*SPS : SPS][MEMORY_DEPTH:]
    
    def evm(ideal, test):
        scaled = test * (np.mean(np.abs(ideal)) / np.mean(np.abs(test)))
        return 10 * np.log10(np.mean(np.abs(ideal - scaled)**2) / np.mean(np.abs(ideal)**2))
    
    evm_raw, evm_poly, evm_nn = evm(sym_ideal, sym_distorted), evm(sym_ideal, sym_poly), evm(sym_ideal, sym_nn)
    
    # Generate Spectra (ILA)
    X_ideal_poly = np.column_stack([x_ideal, x_ideal * (np.abs(x_ideal)**2), x_ideal * (np.abs(x_ideal)**4), x_ideal * (np.abs(x_ideal)**6)])
    pa_lin_poly = simulate_pa(np.dot(X_ideal_poly, poly_coeffs), SPS)
    
    X_ideal_nn_full, _ = create_volterra_dataset(x_ideal, x_ideal, MEMORY_DEPTH)
    nn_dpd_raw_ila = nn_model.predict(X_ideal_nn_full, batch_size=active_batch_size, verbose=0)
    x_dpd_nn_ila = np.zeros_like(x_ideal, dtype=complex)
    x_dpd_nn_ila[MEMORY_DEPTH : MEMORY_DEPTH + len(nn_dpd_raw_ila)] = nn_dpd_raw_ila[:, 0] + 1j * nn_dpd_raw_ila[:, 1]
    pa_lin_nn = simulate_pa(x_dpd_nn_ila, SPS)
    
    # Run Two Tone IM3 Tests
    pin_db, pout_fund, pout_im3, df_sweep, im3_l_arr, im3_u_arr = analyze_im3_and_memory()
    
    progress_bar.progress(100)
    status_text.text("Simulation Complete!")

    # ==========================================
    # UI METRICS AND GRAPHS
    # ==========================================
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Raw PA EVM (No DPD)", f"{evm_raw:.2f} dB")
    col2.metric("Polynomial DPD EVM", f"{evm_poly:.2f} dB", f"{evm_poly - evm_raw:.2f} dB", delta_color="inverse")
    col3.metric("Volterra NN DPD EVM", f"{evm_nn:.2f} dB", f"{evm_nn - evm_raw:.2f} dB", delta_color="inverse")

    tab1, tab2, tab3 = st.tabs(["Spectrum & Constellations", "AM-AM & IM3 Asymptotes", "Memory Asymmetry (IM3 Freq Sweep)"])

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
        ax2.scatter(sym_distorted[:subset].real, sym_distorted[:subset].imag, color='r', s=2, alpha=0.4, label='Raw')
        ax2.scatter(sym_nn[:subset].real, sym_nn[:subset].imag, color='g', s=2, alpha=0.8, label='NN Corrected')
        ax2.set_title(f'Recovered Constellation ({QAM_LEVEL}-QAM)')
        ax2.legend()
        ax2.set_aspect('equal')
        
        st.pyplot(fig1)

    with tab2:
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
        
        # AM-AM Linearization Plot
        ax3.scatter(np.abs(x_ideal[::10]), np.abs(y_distorted[::10]), s=1, alpha=0.3, label='Raw PA', color='red')
        ax3.scatter(np.abs(x_ideal[::10]), np.abs(pa_lin_nn[::10]), s=1, alpha=0.3, label='NN Linearized', color='green')
        ax3.set_title('AM-AM Hardware Linearization')
        ax3.set_xlabel('Input Amplitude |x|')
        ax3.set_ylabel('Output Amplitude |y|')
        ax3.legend()
        ax3.grid(True)
        
        # Pout vs Pin (IM3 Asymptote)
        ax4.plot(pin_db, pout_fund, 'bo-', label='Fundamental')
        ax4.plot(pin_db, pout_im3, 'ro-', label='IM3 Product')
        
        # Draw theoretical asymptotes using the linear region (first 3 points)
        y_int_fund = np.mean(np.array(pout_fund[:3]) - 1.0 * np.array(pin_db[:3]))
        y_int_im3 = np.mean(np.array(pout_im3[:3]) - 3.0 * np.array(pin_db[:3]))
        
        extrapolate_x = np.linspace(np.min(pin_db)-5, np.max(pin_db)+10, 10)
        ax4.plot(extrapolate_x, 1.0 * extrapolate_x + y_int_fund, 'b--', alpha=0.5, label='Slope 1:1')
        ax4.plot(extrapolate_x, 3.0 * extrapolate_x + y_int_im3, 'r--', alpha=0.5, label='Slope 3:1 (IM3)')
        
        ax4.set_title('Pout vs Pin (IP3 Extrapolation)')
        ax4.set_xlabel('Input Power (dB)')
        ax4.set_ylabel('Output Power (dB)')
        ax4.set_ylim(np.min(pout_im3)-5, np.max(pout_fund)+15)
        ax4.legend()
        ax4.grid(True)
        
        st.pyplot(fig2)
        
    with tab3:
        fig3, ax5 = plt.subplots(figsize=(10, 6))
        
        # Memory Asymmetry Plot (Upper vs Lower IM3)
        ax5.plot(df_sweep, im3_l_arr, 'r-', linewidth=2, label='Lower IM3 (-3Δf)')
        ax5.plot(df_sweep, im3_u_arr, 'b-', linewidth=2, label='Upper IM3 (+3Δf)')
        ax5.fill_between(df_sweep, im3_l_arr, im3_u_arr, color='gray', alpha=0.2, label='Memory-Induced Asymmetry')
        
        ax5.set_title('IM3 Asymmetry vs. Tone Spacing (Proof of PA Memory)')
        ax5.set_xlabel('Tone Spacing Δf (Normalized)')
        ax5.set_ylabel('IM3 Power (dB)')
        ax5.legend()
        ax5.grid(True)
        
        st.pyplot(fig3)
