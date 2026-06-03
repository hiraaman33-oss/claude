"""
Scientific schematic: HaCaT genomic DNA adsorbed on disordered Ag/SiNW substrate.
Style: publication-quality, similar to Mussi et al. Micromachines 2021 / Mater.Sci.Eng.C 2021.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, FancyBboxPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import random

random.seed(42)
np.random.seed(42)

fig = plt.figure(figsize=(12, 9), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')

# ── Background gradient (simulating lab/microscope feel) ─────────────────────
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((0, 0), 12, 9, color='#f8f9fc', zorder=0))

# ── Silicon substrate base ────────────────────────────────────────────────────
substrate_y = 1.5
ax.add_patch(FancyBboxPatch((0.5, 0.2), 11, substrate_y - 0.2,
    boxstyle="round,pad=0.05", facecolor='#8B9DC3', edgecolor='#4A5568', lw=1.5, zorder=1))
ax.text(6.0, 0.75, 'Si substrate', ha='center', va='center',
        fontsize=11, color='white', fontweight='bold', zorder=2)

# ── Draw disordered Ag/SiNW nanowires ────────────────────────────────────────
# Each NW: a thick line (Si core) with Ag coating (silver outline)
nw_data = []  # store for hotspot drawing

# Generate random nanowires sprouting from substrate
n_wires = 38
base_xs = np.linspace(0.8, 11.2, n_wires) + np.random.uniform(-0.25, 0.25, n_wires)

for i, bx in enumerate(base_xs):
    length  = np.random.uniform(2.2, 4.8)
    angle   = np.random.uniform(55, 125)   # mostly upward, disordered
    angle_r = np.radians(angle)
    ex = bx + length * np.cos(angle_r)
    ey = substrate_y + length * np.sin(angle_r)
    ey = min(ey, 7.8)
    # Si core
    ax.plot([bx, ex], [substrate_y, ey],
            color='#6B7280', lw=2.8, solid_capstyle='round', zorder=3)
    # Ag coating (silver sheen)
    ax.plot([bx, ex], [substrate_y, ey],
            color='#D1D5DB', lw=1.2, alpha=0.7, solid_capstyle='round', zorder=4)
    nw_data.append(((bx, substrate_y), (ex, ey)))

# Highlight a few leaning/crossed NWs (coffee-ring effect at drop edge)
for i in range(6):
    bx = np.random.uniform(2.5, 9.5)
    angle_r = np.radians(np.random.uniform(15, 45))
    length  = np.random.uniform(1.5, 3.0)
    ex = bx + length * np.cos(angle_r)
    ey = substrate_y + length * np.sin(angle_r)
    ax.plot([bx, ex], [substrate_y, ey],
            color='#9CA3AF', lw=2.2, solid_capstyle='round', zorder=3, alpha=0.8)
    ax.plot([bx, ex], [substrate_y, ey],
            color='#E5E7EB', lw=1.0, alpha=0.6, solid_capstyle='round', zorder=4)
    nw_data.append(((bx, substrate_y), (ex, ey)))

# ── SERS hotspots at nanowire junctions ──────────────────────────────────────
hotspot_positions = [
    (3.2, 3.1), (5.7, 2.9), (7.4, 3.5), (4.5, 4.2),
    (6.8, 4.8), (2.8, 4.0), (8.5, 3.2), (5.1, 5.1),
]
for hx, hy in hotspot_positions:
    glow = plt.Circle((hx, hy), 0.32, color='#FBBF24', alpha=0.18, zorder=5)
    core = plt.Circle((hx, hy), 0.14, color='#F59E0B', alpha=0.55, zorder=6)
    ax.add_patch(glow)
    ax.add_patch(core)

# ── Draw DNA double helix on nanowires ────────────────────────────────────────
def draw_dna_helix(ax, x0, y0, x1, y1, n_turns=3.5, zorder=10):
    """Draw a simplified double helix along the nanowire direction."""
    dx = x1 - x0
    dy = y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    if length < 0.1:
        return
    ux, uy = dx / length, dy / length   # unit along wire
    px, py = -uy, ux                    # perpendicular

    t = np.linspace(0, 1, 200)
    angle = 2 * np.pi * n_turns * t

    amp = 0.18   # helix amplitude

    # Strand 1
    s1x = x0 + t * dx + amp * np.cos(angle) * px
    s1y = y0 + t * dy + amp * np.cos(angle) * py
    # Strand 2 (180° offset)
    s2x = x0 + t * dx + amp * np.cos(angle + np.pi) * px
    s2y = y0 + t * dy + amp * np.cos(angle + np.pi) * py

    ax.plot(s1x, s1y, color='#2563EB', lw=1.8, zorder=zorder, solid_capstyle='round')
    ax.plot(s2x, s2y, color='#DC2626', lw=1.8, zorder=zorder, solid_capstyle='round')

    # Base pairs (rungs) — draw only where strands cross the midline (cos≈0)
    crossings = np.where(np.diff(np.sign(np.cos(angle))))[0]
    for ci in crossings:
        bx1, by1 = s1x[ci], s1y[ci]
        bx2, by2 = s2x[ci], s2y[ci]
        ax.plot([bx1, bx2], [by1, by2], color='#6B7280', lw=0.9,
                alpha=0.7, zorder=zorder - 1)

# Place DNA helices on selected nanowires
dna_wire_indices = [4, 8, 14, 20, 26, 31]
for idx in dna_wire_indices:
    if idx < len(nw_data):
        (bx, by), (ex, ey) = nw_data[idx]
        # Place helix along bottom 60% of wire
        mx = bx + 0.6 * (ex - bx)
        my = by + 0.6 * (ey - by)
        draw_dna_helix(ax, bx + 0.15*(ex-bx), by + 0.15*(ey-by), mx, my,
                       n_turns=3, zorder=10)

# ── Ag nanoparticle decorations on NW tips ────────────────────────────────────
for i, ((bx, by), (ex, ey)) in enumerate(nw_data[:n_wires:3]):
    ax.add_patch(plt.Circle((ex, ey), 0.10,
                 color='#C0C0C0', ec='#A0A0A0', lw=0.5, zorder=8))

# ── Drop droplet (transparent) ───────────────────────────────────────────────
from matplotlib.patches import Ellipse
drop = Ellipse((6.0, 4.5), width=7.5, height=5.5,
               facecolor='#BFDBFE', alpha=0.10,
               edgecolor='#93C5FD', lw=1.5, linestyle='--', zorder=2)
ax.add_patch(drop)
ax.text(9.8, 6.9, '5 µL drop\n(20 ng/µL\nHaCaT DNA)',
        ha='center', va='center', fontsize=8.5, color='#1E40AF',
        fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.3', fc='#EFF6FF', ec='#93C5FD', lw=1))

# Arrow pointing to drop
ax.annotate('', xy=(8.8, 6.2), xytext=(9.5, 6.6),
            arrowprops=dict(arrowstyle='->', color='#3B82F6', lw=1.3))

# ── Labels ────────────────────────────────────────────────────────────────────
# DNA helix label
ax.annotate('Genomic DNA\n(double helix)',
            xy=(nw_data[14][0][0]+0.4, nw_data[14][0][1]+2.0),
            xytext=(1.0, 6.5),
            fontsize=9, color='#1E3A8A', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#1E3A8A', lw=1.2,
                            connectionstyle='arc3,rad=0.2'))

# Ag/SiNW label
ax.annotate('Ag/SiNW\n(disordered array)',
            xy=(9.5, substrate_y + 0.6),
            xytext=(9.8, 5.5),
            fontsize=9, color='#374151', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#374151', lw=1.2,
                            connectionstyle='arc3,rad=-0.25'))

# Hotspot label
ax.annotate('SERS hotspot\n(nanogap)',
            xy=(5.7, 2.9), xytext=(3.5, 1.9),
            fontsize=8.5, color='#92400E',
            arrowprops=dict(arrowstyle='->', color='#D97706', lw=1.1,
                            connectionstyle='arc3,rad=0.3'))

# Ag NP label
ax.annotate('Ag nanocoating\n(100 nm)',
            xy=(nw_data[3][1][0], nw_data[3][1][1]),
            xytext=(0.9, 7.5),
            fontsize=8.5, color='#6B7280',
            arrowprops=dict(arrowstyle='->', color='#9CA3AF', lw=1.0,
                            connectionstyle='arc3,rad=0.15'))

# Ag-N bond label
ax.annotate('Ag–N bond\n(~230 cm⁻¹)',
            xy=(hotspot_positions[1][0], hotspot_positions[1][1] + 0.15),
            xytext=(6.8, 2.0),
            fontsize=8.2, color='#B45309',
            arrowprops=dict(arrowstyle='->', color='#D97706', lw=1.0,
                            connectionstyle='arc3,rad=-0.2'))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_x, legend_y = 0.6, 8.7
legend_items = [
    (plt.Line2D([0],[0], color='#2563EB', lw=2), "DNA strand 1"),
    (plt.Line2D([0],[0], color='#DC2626', lw=2), "DNA strand 2"),
    (plt.Line2D([0],[0], color='#6B7280', lw=1.5), "Base pairs"),
    (plt.Line2D([0],[0], color='#D1D5DB', lw=3), "Ag/SiNW"),
    (mpatches.Patch(facecolor='#FBBF24', alpha=0.6), "SERS hotspot"),
]
handles, labels = zip(*legend_items)
legend = ax.legend(handles, labels, loc='upper left',
                   bbox_to_anchor=(0.0, 1.0),
                   fontsize=8, framealpha=0.9,
                   edgecolor='#D1D5DB', fancybox=True,
                   ncol=5, handlelength=1.5)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(6.0, 8.55,
        'HaCaT Genomic DNA on Disordered Ag/SiNW SERS Substrate',
        ha='center', va='center', fontsize=13, fontweight='bold',
        color='#111827')
ax.text(6.0, 8.25,
        '532 nm excitation  |  20 ng/µL  |  5 µL drop-cast  |  100 nm Ag coating  |  PECVD SiNW',
        ha='center', va='center', fontsize=8, color='#6B7280', fontstyle='italic')

# ── Scale bar ─────────────────────────────────────────────────────────────────
ax.plot([10.2, 11.2], [1.85, 1.85], color='black', lw=2)
ax.text(10.7, 2.0, '1 µm', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.savefig('/home/user/claude/AgSiNW_DNA_HaCaT_schematic.png',
            dpi=300, bbox_inches='tight', facecolor='white',
            pad_inches=0.1)
print("Saved: AgSiNW_DNA_HaCaT_schematic.png")
