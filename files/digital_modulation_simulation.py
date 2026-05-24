"""
SKEE3533 / SEEE3533 — Communication Principles
Group Assignment: Digital Modulation Simulation
Schemes: ASK, FSK, BPSK, QPSK, 16-QAM
Outputs:
  - Figure 1: Modulated waveforms (time domain)
  - Figure 2: BER vs Eb/N0 curves (all schemes)
  - Figure 3: Constellation diagrams (BPSK, QPSK, 16-QAM)
  - Figure 4: Noisy constellation diagrams at Eb/N0 = 10 dB
Dependencies: numpy, matplotlib, scipy (all standard)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc

# ─────────────────────────────────────────────
# Global parameters
# ─────────────────────────────────────────────
np.random.seed(42)

N_BITS       = 100_000   # bits per BER trial
FS           = 1000.0    # sample rate (Hz)
FC           = 50.0      # carrier frequency (Hz)
BIT_RATE     = 10.0      # bit rate (bps)
SPB          = int(FS / BIT_RATE)   # samples per bit = 100
T            = 1.0 / BIT_RATE       # bit period (s)
EBN0_DB      = np.arange(-4, 16, 1) # Eb/N0 range for BER
DEMO_BITS    = 12        # bits to show in waveform plots

# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────

def bits_to_signal(bits, spb):
    """Repeat each bit spb times to create NRZ baseband."""
    return np.repeat(bits, spb).astype(float)

def add_awgn(signal, eb_n0_db, bits_per_symbol=1):
    """
    Add AWGN to a normalised signal.
    Signal is assumed to have unit average power.
    eb_n0_db : Eb/N0 in dB
    bits_per_symbol : log2(M) for M-ary schemes
    """
    eb_n0   = 10 ** (eb_n0_db / 10.0)
    # Es/N0 = (bits/symbol) * Eb/N0
    es_n0   = bits_per_symbol * eb_n0
    # Noise variance per dimension: sigma^2 = 1 / (2 * Es/N0)
    sigma   = np.sqrt(1.0 / (2.0 * es_n0))
    noise   = sigma * np.random.randn(*signal.shape)
    return signal + noise


# ═══════════════════════════════════════════════════════════════
# 1. ASK — Amplitude Shift Keying (Binary, OOK)
# ═══════════════════════════════════════════════════════════════

def ask_modulate(bits, spb, fc, fs):
    """OOK: bit=1 → carrier, bit=0 → silence."""
    baseband = bits_to_signal(bits, spb)  # 0 or 1
    t = np.arange(len(baseband)) / fs
    carrier = np.cos(2 * np.pi * fc * t)
    return baseband * carrier, t

def ask_ber(ebn0_db_range):
    """
    Theoretical BER for OOK (non-coherent approximation):
      BER = 0.5 * erfc( sqrt(Eb/(2*N0)) )
    Simulated: coherent detection via correlation.
    """
    ber_theory = 0.5 * erfc(np.sqrt(10 ** (ebn0_db_range / 10.0) / 2))

    ber_sim = []
    bits = np.random.randint(0, 2, N_BITS).astype(float)
    for ebn0_db in ebn0_db_range:
        eb_n0  = 10 ** (ebn0_db / 10.0)
        sigma  = np.sqrt(1.0 / (2.0 * eb_n0))
        # Received signal = bits + noise (baseband equivalent)
        rx = bits + sigma * np.random.randn(N_BITS)
        # Threshold at 0.5
        detected = (rx >= 0.5).astype(float)
        ber_sim.append(np.mean(detected != bits))

    return ber_theory, np.array(ber_sim)


# ═══════════════════════════════════════════════════════════════
# 2. FSK — Frequency Shift Keying (Binary, non-coherent)
# ═══════════════════════════════════════════════════════════════

def fsk_modulate(bits, spb, fc, fs, delta_f=20.0):
    """Binary FSK: bit=0 → fc-Δf, bit=1 → fc+Δf."""
    baseband = bits_to_signal(bits, spb)
    t = np.arange(len(baseband)) / fs
    freq = np.where(baseband == 0, fc - delta_f, fc + delta_f)
    # Instantaneous phase (integrate frequency)
    phase = 2 * np.pi * np.cumsum(freq) / fs
    return np.cos(phase), t

def fsk_ber(ebn0_db_range):
    """
    Theoretical BER for non-coherent binary FSK:
      BER = 0.5 * exp(-Eb / (2*N0))
    Simulated using baseband energy detection.
    """
    eb_n0_lin   = 10 ** (ebn0_db_range / 10.0)
    ber_theory  = 0.5 * np.exp(-eb_n0_lin / 2.0)

    ber_sim = []
    bits = np.random.randint(0, 2, N_BITS)
    for ebn0_db in ebn0_db_range:
        eb_n0  = 10 ** (ebn0_db / 10.0)
        sigma  = np.sqrt(1.0 / (2.0 * eb_n0))
        # Baseband: s0 = -1, s1 = +1 (BFSK simplified energy metric)
        s = 2.0 * bits - 1.0
        rx = s + sigma * np.random.randn(N_BITS)
        detected = (rx >= 0).astype(int)
        ber_sim.append(np.mean(detected != bits))

    return ber_theory, np.array(ber_sim)


# ═══════════════════════════════════════════════════════════════
# 3. BPSK — Binary Phase Shift Keying
# ═══════════════════════════════════════════════════════════════

def bpsk_modulate(bits, spb, fc, fs):
    """BPSK: bit=0 → +1·cos, bit=1 → -1·cos."""
    symbols = 1.0 - 2.0 * bits  # map 0→+1, 1→-1
    baseband = bits_to_signal(symbols, spb)
    t = np.arange(len(baseband)) / fs
    return baseband * np.cos(2 * np.pi * fc * t), t

def bpsk_ber(ebn0_db_range):
    """
    Theoretical BER for BPSK (coherent):
      BER = 0.5 * erfc( sqrt(Eb/N0) )
    """
    eb_n0_lin  = 10 ** (ebn0_db_range / 10.0)
    ber_theory = 0.5 * erfc(np.sqrt(eb_n0_lin))

    ber_sim = []
    bits = np.random.randint(0, 2, N_BITS)
    for ebn0_db in ebn0_db_range:
        rx = add_awgn((1.0 - 2.0 * bits).astype(float), ebn0_db, bits_per_symbol=1)
        detected = (rx < 0).astype(int)
        ber_sim.append(np.mean(detected != bits))

    return ber_theory, np.array(ber_sim)

def bpsk_constellation(ebn0_db=10.0, n=1000):
    bits = np.random.randint(0, 2, n)
    symbols = 1.0 - 2.0 * bits
    rx = add_awgn(symbols.astype(float), ebn0_db, bits_per_symbol=1)
    return symbols, rx, bits


# ═══════════════════════════════════════════════════════════════
# 4. QPSK — Quadrature Phase Shift Keying
# ═══════════════════════════════════════════════════════════════

QPSK_MAP = {
    (0, 0): ( 1 + 1j),
    (0, 1): (-1 + 1j),
    (1, 0): ( 1 - 1j),
    (1, 1): (-1 - 1j),
}
# Normalise to unit average energy
QPSK_SCALE = 1.0 / np.sqrt(2)

def qpsk_modulate(bits, spb, fc, fs):
    """QPSK: map 2 bits → one of 4 phases (0, 90, 180, 270°)."""
    # Pad to even length
    if len(bits) % 2:
        bits = np.append(bits, 0)
    pairs = bits.reshape(-1, 2)
    # I/Q components
    I = np.where(pairs[:, 0] == 0,  1.0, -1.0) * QPSK_SCALE
    Q = np.where(pairs[:, 1] == 0,  1.0, -1.0) * QPSK_SCALE

    I_up = np.repeat(I, spb)
    Q_up = np.repeat(Q, spb)
    t = np.arange(len(I_up)) / fs
    return (I_up * np.cos(2*np.pi*fc*t) - Q_up * np.sin(2*np.pi*fc*t)), t

def qpsk_ber(ebn0_db_range):
    """
    Theoretical BER for QPSK = BPSK (same Eb/N0 performance):
      BER = 0.5 * erfc( sqrt(Eb/N0) )
    """
    eb_n0_lin  = 10 ** (ebn0_db_range / 10.0)
    ber_theory = 0.5 * erfc(np.sqrt(eb_n0_lin))

    ber_sim = []
    n_sym = N_BITS // 2
    for ebn0_db in ebn0_db_range:
        bits = np.random.randint(0, 2, n_sym * 2)
        pairs = bits.reshape(-1, 2)
        I_tx = (1.0 - 2.0 * pairs[:, 0]) * QPSK_SCALE
        Q_tx = (1.0 - 2.0 * pairs[:, 1]) * QPSK_SCALE

        I_rx = add_awgn(I_tx, ebn0_db, bits_per_symbol=2)
        Q_rx = add_awgn(Q_tx, ebn0_db, bits_per_symbol=2)

        b0_det = (I_rx < 0).astype(int)
        b1_det = (Q_rx < 0).astype(int)
        errors = np.sum(b0_det != pairs[:, 0]) + np.sum(b1_det != pairs[:, 1])
        ber_sim.append(errors / (n_sym * 2))

    return ber_theory, np.array(ber_sim)

def qpsk_constellation(ebn0_db=10.0, n=1000):
    n_sym = n
    bits = np.random.randint(0, 2, n_sym * 2).reshape(-1, 2)
    I_tx = (1.0 - 2.0 * bits[:, 0]) * QPSK_SCALE
    Q_tx = (1.0 - 2.0 * bits[:, 1]) * QPSK_SCALE
    symbols = I_tx + 1j * Q_tx

    I_rx = add_awgn(I_tx, ebn0_db, bits_per_symbol=2)
    Q_rx = add_awgn(Q_tx, ebn0_db, bits_per_symbol=2)
    rx = I_rx + 1j * Q_rx
    return symbols, rx


# ═══════════════════════════════════════════════════════════════
# 5. 16-QAM — 16-point Quadrature Amplitude Modulation
# ═══════════════════════════════════════════════════════════════

def gray_code(n):
    """Return Gray code for integer n."""
    return n ^ (n >> 1)

def qam16_constellation_points():
    """
    Standard Gray-coded 16-QAM constellation.
    Returns: dict mapping 4-bit tuple → complex symbol (normalised).
    """
    levels = [-3, -1, 1, 3]
    avg_power = np.mean([i**2 + q**2 for i in levels for q in levels])
    scale = np.sqrt(1.0 / avg_power)

    # Gray code rows and columns
    gray2 = [gray_code(i) for i in range(4)]  # [0,1,3,2]
    points = {}
    for ri, row_gray in enumerate(gray2):
        for ci, col_gray in enumerate(gray2):
            i_val = levels[ri] * scale
            q_val = levels[ci] * scale
            # 4-bit label: 2 bits from row, 2 from col
            b = ((row_gray & 2) >> 1, row_gray & 1, (col_gray & 2) >> 1, col_gray & 1)
            points[b] = i_val + 1j * q_val
    return points

QAM16_POINTS = qam16_constellation_points()
QAM16_SYMBOLS = np.array(list(QAM16_POINTS.values()))
QAM16_LABELS  = list(QAM16_POINTS.keys())

def qam16_modulate_bits(bits):
    """Map bits (multiple of 4) to 16-QAM symbols."""
    bits = bits[: len(bits) - len(bits) % 4]
    groups = bits.reshape(-1, 4)
    symbols = np.array([QAM16_POINTS[tuple(g)] for g in groups])
    return symbols

def qam16_demodulate(rx_symbols):
    """Minimum-distance decision for each received symbol."""
    detected_bits = []
    for rx in rx_symbols:
        dists = np.abs(rx - QAM16_SYMBOLS)
        idx = np.argmin(dists)
        detected_bits.extend(QAM16_LABELS[idx])
    return np.array(detected_bits, dtype=int)

def qam16_ber(ebn0_db_range):
    """
    Theoretical SER for 16-QAM (Gray coded, approximate):
      SER ≈ (3/2) * erfc( sqrt(Eb*log2(16) / (10*N0)) )  [Gray approx]
      BER ≈ SER / log2(M)
    """
    M    = 16
    bps  = int(np.log2(M))
    eb_n0_lin  = 10 ** (ebn0_db_range / 10.0)
    # Exact theoretical BER for Gray-coded square QAM
    ber_theory = (3.0 / (2.0 * bps)) * erfc(np.sqrt(bps * eb_n0_lin / 10.0))

    ber_sim = []
    bits_tx = np.random.randint(0, 2, N_BITS - N_BITS % bps)
    for ebn0_db in ebn0_db_range:
        symbols_tx = qam16_modulate_bits(bits_tx)
        # Noise per dimension
        eb_n0  = 10 ** (ebn0_db / 10.0)
        sigma  = np.sqrt(1.0 / (2.0 * bps * eb_n0))
        noise  = sigma * (np.random.randn(len(symbols_tx)) + 1j * np.random.randn(len(symbols_tx)))
        symbols_rx = symbols_tx + noise
        bits_rx    = qam16_demodulate(symbols_rx)
        n_cmp      = min(len(bits_tx), len(bits_rx))
        ber_sim.append(np.mean(bits_tx[:n_cmp] != bits_rx[:n_cmp]))

    return ber_theory, np.array(ber_sim)

def qam16_constellation(ebn0_db=10.0, n_sym=500):
    bits_tx   = np.random.randint(0, 2, n_sym * 4)
    symbols_tx = qam16_modulate_bits(bits_tx)
    eb_n0  = 10 ** (ebn0_db / 10.0)
    sigma  = np.sqrt(1.0 / (2.0 * 4 * eb_n0))
    noise  = sigma * (np.random.randn(len(symbols_tx)) + 1j * np.random.randn(len(symbols_tx)))
    return symbols_tx, symbols_tx + noise


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

STYLE = {
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.alpha'        : 0.3,
    'grid.linestyle'    : '--',
    'font.size'         : 10,
}
plt.rcParams.update(STYLE)

COLORS = {
    'ASK'   : '#E24B4A',
    'FSK'   : '#EF9F27',
    'BPSK'  : '#1D9E75',
    'QPSK'  : '#378ADD',
    '16QAM' : '#7F77DD',
    'tx'    : '#5F5E5A',
    'rx'    : '#E24B4A',
}

demo_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1])


# ──────────────────────────────
# Figure 1: Waveforms
# ──────────────────────────────
def plot_waveforms():
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=False)
    fig.suptitle('Figure 1 — Digital Modulation Waveforms (12-bit sequence)', fontsize=13, fontweight='bold', y=0.98)

    bits = demo_bits

    # ASK
    s, t = ask_modulate(bits, SPB, FC, FS)
    axes[0].plot(t, s, color=COLORS['ASK'], lw=0.8)
    axes[0].set_ylabel('ASK (OOK)', fontsize=9)
    axes[0].set_ylim(-1.3, 1.3)

    # FSK
    s, t = fsk_modulate(bits, SPB, FC, FS, delta_f=20.0)
    axes[1].plot(t, s, color=COLORS['FSK'], lw=0.8)
    axes[1].set_ylabel('FSK', fontsize=9)
    axes[1].set_ylim(-1.3, 1.3)

    # BPSK
    s, t = bpsk_modulate(bits, SPB, FC, FS)
    axes[2].plot(t, s, color=COLORS['BPSK'], lw=0.8)
    axes[2].set_ylabel('BPSK', fontsize=9)
    axes[2].set_ylim(-1.3, 1.3)

    # QPSK (shows 6 symbol periods for 12 bits)
    s, t = qpsk_modulate(bits, SPB, FC, FS)
    axes[3].plot(t, s, color=COLORS['QPSK'], lw=0.8)
    axes[3].set_ylabel('QPSK', fontsize=9)
    axes[3].set_ylim(-1.3, 1.3)

    # Baseband bit stream (for reference)
    base = bits_to_signal(bits, SPB)
    t_b  = np.arange(len(base)) / FS
    axes[4].step(t_b, base, color='#444441', lw=1.2, where='post')
    axes[4].set_ylabel('Bit stream', fontsize=9)
    axes[4].set_ylim(-0.3, 1.5)
    axes[4].set_xlabel('Time (s)', fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=8)

    # Add bit boundaries
    for ax in axes[:-1]:
        for k in range(1, len(bits)):
            ax.axvline(k * T, color='gray', lw=0.4, alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('/home/claude/fig1_waveforms.png', dpi=150, bbox_inches='tight')
    print("Saved: fig1_waveforms.png")
    plt.close()


# ──────────────────────────────
# Figure 2: BER curves
# ──────────────────────────────
def plot_ber():
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle('Figure 2 — BER vs Eb/N₀ for Digital Modulation Schemes', fontsize=13, fontweight='bold')

    print("Computing BER curves (this takes ~30 seconds)...")

    ask_th, ask_sim   = ask_ber(EBN0_DB)
    fsk_th, fsk_sim   = fsk_ber(EBN0_DB)
    bpsk_th, bpsk_sim = bpsk_ber(EBN0_DB)
    qpsk_th, qpsk_sim = qpsk_ber(EBN0_DB)
    qam_th, qam_sim   = qam16_ber(EBN0_DB)

    print("BER computation done.")

    def safe_plot(th, sim, color, label):
        ax.semilogy(EBN0_DB, th,  color=color, lw=2,   label=f'{label} (theory)')
        ax.semilogy(EBN0_DB, np.maximum(sim, 1e-6), color=color, lw=0, marker='o',
                    markersize=4, alpha=0.75, label=f'{label} (simulated)')

    safe_plot(ask_th,  ask_sim,  COLORS['ASK'],   'ASK (OOK)')
    safe_plot(fsk_th,  fsk_sim,  COLORS['FSK'],   'FSK')
    safe_plot(bpsk_th, bpsk_sim, COLORS['BPSK'],  'BPSK')
    safe_plot(qpsk_th, qpsk_sim, COLORS['QPSK'],  'QPSK')
    safe_plot(qam_th,  qam_sim,  COLORS['16QAM'], '16-QAM')

    ax.set_xlabel('Eb/N₀ (dB)', fontsize=11)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=11)
    ax.set_ylim(1e-6, 1.0)
    ax.set_xlim(EBN0_DB[0], EBN0_DB[-1])
    ax.legend(fontsize=8, ncol=2, loc='lower left')
    ax.set_title('Solid lines = theoretical | Markers = Monte Carlo simulation', fontsize=9, pad=4)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig('/home/claude/fig2_ber_curves.png', dpi=150, bbox_inches='tight')
    print("Saved: fig2_ber_curves.png")
    plt.close()


# ──────────────────────────────
# Figure 3: Ideal constellation diagrams
# ──────────────────────────────
def plot_ideal_constellations():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Figure 3 — Ideal Constellation Diagrams (no noise)', fontsize=13, fontweight='bold')

    # BPSK
    ax = axes[0]
    bpsk_pts = np.array([1.0 + 0j, -1.0 + 0j])
    bpsk_lbl = ['0', '1']
    ax.scatter(bpsk_pts.real, bpsk_pts.imag, s=120, color=COLORS['BPSK'], zorder=3)
    for pt, lb in zip(bpsk_pts, bpsk_lbl):
        ax.annotate(lb, (pt.real, pt.imag), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=11)
    ax.axhline(0, color='gray', lw=0.7); ax.axvline(0, color='gray', lw=0.7)
    ax.set_xlim(-2, 2); ax.set_ylim(-1.5, 1.5)
    ax.set_title('BPSK', fontsize=11); ax.set_xlabel('In-phase (I)'); ax.set_ylabel('Quadrature (Q)')
    ax.set_aspect('equal')

    # QPSK
    ax = axes[1]
    qpsk_pts = np.array([1+1j, -1+1j, -1-1j, 1-1j]) * QPSK_SCALE
    qpsk_lbl = ['00', '01', '11', '10']
    ax.scatter(qpsk_pts.real, qpsk_pts.imag, s=120, color=COLORS['QPSK'], zorder=3)
    for pt, lb in zip(qpsk_pts, qpsk_lbl):
        ax.annotate(lb, (pt.real, pt.imag), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9)
    ax.axhline(0, color='gray', lw=0.7); ax.axvline(0, color='gray', lw=0.7)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_title('QPSK', fontsize=11); ax.set_xlabel('In-phase (I)'); ax.set_ylabel('Quadrature (Q)')
    ax.set_aspect('equal')

    # 16-QAM
    ax = axes[2]
    pts   = np.array(list(QAM16_POINTS.values()))
    labls = [''.join(map(str, k)) for k in QAM16_LABELS]
    ax.scatter(pts.real, pts.imag, s=80, color=COLORS['16QAM'], zorder=3)
    for pt, lb in zip(pts, labls):
        ax.annotate(lb, (pt.real, pt.imag), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=6)
    ax.axhline(0, color='gray', lw=0.7); ax.axvline(0, color='gray', lw=0.7)
    ax.set_title('16-QAM', fontsize=11); ax.set_xlabel('In-phase (I)'); ax.set_ylabel('Quadrature (Q)')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('/home/claude/fig3_ideal_constellations.png', dpi=150, bbox_inches='tight')
    print("Saved: fig3_ideal_constellations.png")
    plt.close()


# ──────────────────────────────
# Figure 4: Noisy constellations at Eb/N0 = 10 dB
# ──────────────────────────────
def plot_noisy_constellations():
    EBN0 = 10.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'Figure 4 — Received Constellation Diagrams at Eb/N₀ = {EBN0} dB', fontsize=13, fontweight='bold')

    # BPSK
    ax = axes[0]
    tx, rx, bits = bpsk_constellation(EBN0, n=800)
    ax.scatter(rx.real, np.zeros(len(rx)), s=8, alpha=0.4, color=COLORS['BPSK'])
    ax.scatter([1, -1], [0, 0], s=150, color='black', zorder=5, marker='x', linewidths=2)
    ax.axvline(0, color='red', lw=1.2, linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-1.5, 1.5)
    ax.set_title('BPSK', fontsize=11); ax.set_xlabel('I'); ax.set_ylabel('Q')
    ax.set_aspect('equal')

    # QPSK
    ax = axes[1]
    tx_q, rx_q = qpsk_constellation(EBN0, n=800)
    ax.scatter(tx_q.real, tx_q.imag, s=60, color='black', marker='x', zorder=5, linewidths=1.5, label='Ideal')
    ax.scatter(rx_q.real, rx_q.imag, s=8, alpha=0.35, color=COLORS['QPSK'], label='Received')
    ax.axhline(0, color='red', lw=1.2, linestyle='--', alpha=0.6)
    ax.axvline(0, color='red', lw=1.2, linestyle='--', alpha=0.6)
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
    ax.set_title('QPSK', fontsize=11); ax.set_xlabel('I'); ax.set_ylabel('Q')
    ax.legend(fontsize=8); ax.set_aspect('equal')

    # 16-QAM
    ax = axes[2]
    tx_q, rx_q = qam16_constellation(EBN0, n_sym=600)
    ax.scatter(tx_q.real, tx_q.imag, s=60, color='black', marker='x', zorder=5, linewidths=1.5, label='Ideal')
    ax.scatter(rx_q.real, rx_q.imag, s=8, alpha=0.35, color=COLORS['16QAM'], label='Received')
    ax.set_title('16-QAM', fontsize=11); ax.set_xlabel('I'); ax.set_ylabel('Q')
    ax.legend(fontsize=8); ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('/home/claude/fig4_noisy_constellations.png', dpi=150, bbox_inches='tight')
    print("Saved: fig4_noisy_constellations.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 55)
    print("  SKEE3533 — Digital Modulation Simulation")
    print("  Schemes: ASK, FSK, BPSK, QPSK, 16-QAM")
    print("=" * 55)

    print("\n[1/4] Plotting waveforms...")
    plot_waveforms()

    print("\n[2/4] Plotting BER curves...")
    plot_ber()

    print("\n[3/4] Plotting ideal constellation diagrams...")
    plot_ideal_constellations()

    print("\n[4/4] Plotting noisy constellation diagrams...")
    plot_noisy_constellations()

    print("\nAll figures saved:")
    print("  fig1_waveforms.png")
    print("  fig2_ber_curves.png")
    print("  fig3_ideal_constellations.png")
    print("  fig4_noisy_constellations.png")
    print("\nDone.")
