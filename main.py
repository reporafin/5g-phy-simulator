import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math

from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import compute_ber, ebnodb2no
from sionna.phy.channel import AWGN
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder

class Professional5GLink(tf.keras.Model):
    def __init__(self, use_ldpc=True, batch_size=128):
        super().__init__()
        self.use_ldpc = use_ldpc
        self.batch_size = batch_size
        self.num_bits_per_symbol = 4 
        self.k = 1000 
        self.n = 2000 
        
        self.source = BinarySource()
        self.mapper = Mapper("qam", self.num_bits_per_symbol)
        self.awgn = AWGN()
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)
        
        if self.use_ldpc:
            self.encoder = LDPC5GEncoder(self.k, self.n)
            self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

    @tf.function
    def call(self, ebno_db):
        coderate = self.k / self.n if self.use_ldpc else 1.0
        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, coderate)
        
        bits = self.source([self.batch_size, self.k])
        
        if self.use_ldpc:
            encoded_bits = self.encoder(bits)
            x = self.mapper(encoded_bits)
        else:
            x = self.mapper(bits)
            
        h_real = tf.random.normal(tf.shape(tf.math.real(x)))
        h_imag = tf.random.normal(tf.shape(tf.math.imag(x)))
        h = tf.complex(h_real, h_imag) / tf.cast(math.sqrt(2.0), tf.complex64)
        
        y_faded = x * h
        y = self.awgn(y_faded, no) 
        
        y_eq = y / h
        no_eff = no / tf.cast((tf.abs(h)**2), tf.float32)
        
        llr = self.demapper(y_eq, no_eff) 
        
        if self.use_ldpc:
            bits_rx = self.decoder(llr)
        else:
            bits_rx = tf.cast(tf.math.greater(llr, 0), tf.float32)
            
        # UPGRADE: Now returning 'y' (Raw Signal) as well
        return bits, bits_rx, x, y_eq, y

def run_simulation():
    print("Initializing 5G PHY Simulation")
    batch_size = 128
    model_uncoded = Professional5GLink(use_ldpc=False, batch_size=batch_size)
    model_coded = Professional5GLink(use_ldpc=True, batch_size=batch_size)
    
    ebno_dbs = np.arange(0, 16, 1.0) 
    ber_uncoded, ber_coded = [], []
    
    plot_tx_symbols = None
    plot_rx_eq_symbols = None
    plot_rx_raw_symbols = None # raw symbols
    
    print("Running Monte Carlo Simulations...")
    for ebno_db in ebno_dbs:
        ebno_tensor = tf.constant(ebno_db, dtype=tf.float32)
        
        #5 variables
        tx_u, rx_u, _, _, _ = model_uncoded(ebno_tensor)
        ber_u = compute_ber(tx_u, rx_u).numpy()
        ber_uncoded.append(ber_u)
        
        tx_c, rx_c, x_c, y_eq_c, y_raw_c = model_coded(ebno_tensor)
        ber_c = compute_ber(tx_c, rx_c).numpy()
        ber_coded.append(ber_c)
        
        # Capture at 15 dB
        if ebno_db == 15.0:
            plot_tx_symbols = tf.reshape(x_c, [-1]).numpy()
            plot_rx_eq_symbols = tf.reshape(y_eq_c, [-1]).numpy()
            plot_rx_raw_symbols = tf.reshape(y_raw_c, [-1]).numpy() #raw signal
            
        print(f"Eb/No: {ebno_db:4.1f} dB | Uncoded BER: {ber_u:.5f} | LDPC BER: {ber_c:.5f}")

    #DASHBOARD
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6)) # Upgraded to 1x3 grid

    # 1 BER Curve
    ax1.semilogy(ebno_dbs, ber_uncoded, 'r--', marker='o', linewidth=2, label='Uncoded 16-QAM')
    ax1.semilogy(ebno_dbs, ber_coded, 'b-', marker='s', linewidth=2, label='3GPP LDPC 16-QAM')
    ax1.set_title("5G Link Performance", fontsize=14)
    ax1.set_xlabel("Eb/No (dB)", fontsize=12)
    ax1.set_ylabel("Bit Error Rate (BER)", fontsize=12)
    ax1.set_ylim(1e-5, 1.0)
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.legend()

    #2 Raw Received Constellation
    ax2.scatter(np.real(plot_rx_raw_symbols[:2000]), np.imag(plot_rx_raw_symbols[:2000]), 
                color='purple', alpha=0.2, s=10, label='Received (Faded & Noisy)')
    ax2.set_title("Raw Signal (Before Equalizer)", fontsize=14)
    ax2.set_xlabel("In-Phase (I)", fontsize=12)
    ax2.set_ylabel("Quadrature (Q)", fontsize=12)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.grid(True, ls="--", alpha=0.7)
    ax2.legend()

    #3 Equalized Constellation
    ax3.scatter(np.real(plot_rx_eq_symbols[:2000]), np.imag(plot_rx_eq_symbols[:2000]), 
                color='blue', alpha=0.2, s=10, label='Received (Equalized)')
    ax3.scatter(np.real(plot_tx_symbols[:16]), np.imag(plot_tx_symbols[:16]), 
                color='red', marker='x', s=100, linewidth=2, label='Transmitted (Ideal)')
    ax3.set_title("Equalized Signal (After Zero-Forcing)", fontsize=14)
    ax3.set_xlabel("In-Phase (I)", fontsize=12)
    ax3.set_ylabel("Quadrature (Q)", fontsize=12)
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.axvline(0, color='black', linewidth=0.5)
    ax3.set_xlim(-2.5, 2.5)
    ax3.set_ylim(-2.5, 2.5)
    ax3.grid(True, ls="--", alpha=0.7)
    ax3.legend()

    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/5g_simulation_dashboard.png', dpi=300, bbox_inches='tight')
    print("\nSimulation complete! Dashboard saved to results/5g_simulation_dashboard.png")
    plt.show()

if __name__ == "__main__":
    run_simulation()