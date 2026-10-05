# 10_ground_ops/link_budget.py
"""RF link budget: FSPL, EIRP, C/N0, Eb/N0, margin (standard textbook formulation)."""
import numpy as np

K_BOLTZMANN_DBW_HZ_K = -228.6


def free_space_path_loss_db(range_km, freq_hz):
    freq_mhz = freq_hz / 1e6
    return 20 * np.log10(range_km) + 20 * np.log10(freq_mhz) + 32.44


def eirp_dbw(tx_power_dbw, tx_antenna_gain_dbi, tx_line_loss_db=0.0):
    return tx_power_dbw + tx_antenna_gain_dbi - tx_line_loss_db


def received_power_dbw(eirp_dbw_val, path_loss_db, rx_antenna_gain_dbi,
                        rx_line_loss_db=0.0, atmospheric_loss_db=0.0, pointing_loss_db=0.0):
    return (eirp_dbw_val - path_loss_db + rx_antenna_gain_dbi
            - rx_line_loss_db - atmospheric_loss_db - pointing_loss_db)


def noise_power_spectral_density_dbw_hz(system_noise_temp_k):
    return K_BOLTZMANN_DBW_HZ_K + 10 * np.log10(system_noise_temp_k)


def carrier_to_noise_density_db(received_power_dbw_val, noise_psd_dbw_hz):
    return received_power_dbw_val - noise_psd_dbw_hz


def eb_n0_db(c_n0_db_hz, data_rate_bps):
    return c_n0_db_hz - 10 * np.log10(data_rate_bps)


def link_margin_db(achieved_eb_n0_db, required_eb_n0_db):
    return achieved_eb_n0_db - required_eb_n0_db


def evaluate_link(tx_power_w, tx_antenna_gain_dbi, tx_line_loss_db,
                   range_km, freq_hz,
                   rx_antenna_gain_dbi, rx_system_noise_temp_k, rx_line_loss_db,
                   data_rate_bps, required_eb_n0_db,
                   atmospheric_loss_db=0.5, pointing_loss_db=1.0):
    tx_power_dbw = 10 * np.log10(tx_power_w)
    eirp = eirp_dbw(tx_power_dbw, tx_antenna_gain_dbi, tx_line_loss_db)
    fspl = free_space_path_loss_db(range_km, freq_hz)
    p_rx = received_power_dbw(eirp, fspl, rx_antenna_gain_dbi, rx_line_loss_db,
                               atmospheric_loss_db, pointing_loss_db)
    n0 = noise_power_spectral_density_dbw_hz(rx_system_noise_temp_k)
    c_n0 = carrier_to_noise_density_db(p_rx, n0)
    ebn0 = eb_n0_db(c_n0, data_rate_bps)
    margin = link_margin_db(ebn0, required_eb_n0_db)
    return {
        "tx_power_dbw": tx_power_dbw, "eirp_dbw": eirp, "fspl_db": fspl,
        "received_power_dbw": p_rx, "noise_psd_dbw_hz": n0, "c_n0_db_hz": c_n0,
        "eb_n0_db": ebn0, "required_eb_n0_db": required_eb_n0_db,
        "margin_db": margin, "link_closes": margin >= 0,
    }
