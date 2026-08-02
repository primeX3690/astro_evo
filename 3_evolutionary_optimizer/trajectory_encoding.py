# 3_evolutionary_optimizer/trajectory_encoding.py
"""Encodes a candidate Lambert-transfer trajectory as a gene: [transfer_angle_deg, tof_hours]."""
import numpy as np

GENE_BOUNDS = {
    "transfer_angle_deg": (90.0, 179.5),  # avoids exact 180 (Lambert A=0 singularity)
    "tof_hours": (2.0, 12.0),
}


def decode_gene(gene, r1_km, r2_km):
    angle_deg, tof_hr = gene
    angle = np.radians(angle_deg)
    r1_vec = np.array([r1_km, 0.0, 0.0])
    r2_vec = np.array([r2_km * np.cos(angle), r2_km * np.sin(angle), 0.0])
    tof_s = tof_hr * 3600.0
    return r1_vec, r2_vec, tof_s


def random_gene(rng):
    a_lo, a_hi = GENE_BOUNDS["transfer_angle_deg"]
    t_lo, t_hi = GENE_BOUNDS["tof_hours"]
    return np.array([rng.uniform(a_lo, a_hi), rng.uniform(t_lo, t_hi)])


def clip_gene(gene):
    a_lo, a_hi = GENE_BOUNDS["transfer_angle_deg"]
    t_lo, t_hi = GENE_BOUNDS["tof_hours"]
    return np.array([np.clip(gene[0], a_lo, a_hi), np.clip(gene[1], t_lo, t_hi)])