"""
Reads LabSpec6 .l6m Raman map files and averages the Y-axis spectra point by point.

Folder: C:\Users\Hira Aman\Downloads\15 days old drop
Each map: 1 X position, 25 Y positions, N spectral (wavenumber) points.
The script averages corresponding Y positions across all 4 maps.
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import glob


# ── LabSpec6 .l6m binary parser ──────────────────────────────────────────────

def _read_tag(f):
    """Read one TLV-style tag from the file. Returns (tag, data) or None at EOF."""
    raw = f.read(4)
    if len(raw) < 4:
        return None
    tag_len = struct.unpack('<I', raw)[0]
    if tag_len == 0 or tag_len > 256:
        return None
    tag = f.read(tag_len).decode('latin-1', errors='replace').rstrip('\x00')
    raw = f.read(4)
    if len(raw) < 4:
        return None
    data_len = struct.unpack('<I', raw)[0]
    data = f.read(data_len)
    return tag, data


def parse_l6m(filepath):
    """
    Parse a LabSpec6 .l6m map file.

    Returns
    -------
    wavenumbers : 1-D ndarray  (N,)
    spectra     : 2-D ndarray  (n_points, N)  – one row per map point (Y positions)
    nx, ny      : grid dimensions
    """
    with open(filepath, 'rb') as f:
        # Signature check
        sig = f.read(10)
        if b'LabSpec6' not in sig:
            raise ValueError(f"Not a LabSpec6 file: {filepath}")
        f.seek(0)
        raw = f.read()

    # Locate the axis data block and spectral data block by scanning for known tags
    # The file uses length-prefixed tag names followed by length-prefixed data.
    # We scan for float32 arrays matching expected sizes.

    # Strategy: find "WaveNum" / "Spectro" / axis blocks then the large intensity block.
    # Fall back to brute-force float32 scan.

    wavenumbers, spectra, nx, ny = _extract_via_known_tags(raw)
    if wavenumbers is None:
        raise RuntimeError(
            "Could not parse the map file. "
            "Make sure the file is a valid LabSpec6 .l6m map."
        )
    return wavenumbers, spectra, nx, ny


def _read_pascal_str(raw, pos):
    """Read a length-prefixed string at pos; return (string, new_pos)."""
    if pos + 4 > len(raw):
        return None, pos
    slen = struct.unpack_from('<I', raw, pos)[0]
    pos += 4
    if slen == 0 or slen > 512 or pos + slen > len(raw):
        return None, pos
    s = raw[pos:pos + slen].decode('latin-1', errors='replace').rstrip('\x00')
    pos += slen
    return s, pos


def _extract_via_known_tags(raw):
    """Walk the TLV structure and pull out grid dims, axis, and spectral data."""
    pos = 0
    nx = ny = None
    wavenumbers = None
    n_wn = None
    spectra_bytes = None

    while pos < len(raw) - 8:
        tag, pos2 = _read_pascal_str(raw, pos)
        if tag is None:
            pos += 1
            continue

        if pos2 + 4 > len(raw):
            break
        dlen = struct.unpack_from('<I', raw, pos2)[0]
        pos2 += 4
        data = raw[pos2:pos2 + dlen]
        next_pos = pos2 + dlen

        tag_clean = tag.strip()

        # Grid size
        if tag_clean in ('Map', 'MapArea', 'SpIm') and dlen >= 8:
            try:
                # Often stores (nx, ny) as two int32s
                nx_c, ny_c = struct.unpack_from('<ii', data, 0)
                if 1 <= nx_c <= 10000 and 1 <= ny_c <= 10000:
                    nx, ny = nx_c, ny_c
            except Exception:
                pass

        # X axis (wavenumber)
        if tag_clean in ('Wnum', 'WaveNum', 'Spectro', 'axis', 'axi0', 'axi1', 'axi{') and dlen >= 8:
            count = dlen // 4
            try:
                arr = np.frombuffer(data, dtype='<f4')
                # Sanity: Raman wavenumbers are roughly 100–4000 cm⁻¹
                if np.all((arr > 50) & (arr < 5000)) and len(arr) > 10:
                    wavenumbers = arr
                    n_wn = len(arr)
            except Exception:
                pass

        # Large float32 block → likely the spectral intensity matrix
        if dlen > 1000 and n_wn and dlen % 4 == 0:
            count = dlen // 4
            if count % n_wn == 0:
                n_pts = count // n_wn
                if 1 <= n_pts <= 10000:
                    spectra_bytes = data
                    n_spectra = n_pts

        pos = next_pos

    # ── Fallback: if we didn't find wavenumbers, brute-force scan for a plausible
    #             Raman wavenumber axis (ascending floats 100–4000).
    if wavenumbers is None:
        wavenumbers, n_wn = _find_wavenumber_axis(raw)

    if wavenumbers is None or spectra_bytes is None:
        return None, None, None, None

    spectra = np.frombuffer(spectra_bytes, dtype='<f4').reshape(-1, n_wn)

    # Grid dims from spectra shape if not found in tags
    if ny is None:
        ny = spectra.shape[0]
        nx = 1

    return wavenumbers, spectra, nx, ny


def _find_wavenumber_axis(raw):
    """Brute-force: find a run of ascending float32s in the Raman range."""
    arr_all = np.frombuffer(raw[:], dtype='<f4')
    for i in range(len(arr_all) - 50):
        window = arr_all[i:i + 50]
        if (np.all(window > 100) and np.all(window < 4500) and
                np.all(np.diff(window) > 0)):
            # Extend the run
            j = i + 50
            while j < len(arr_all) and 100 < arr_all[j] < 4500:
                j += 1
            axis = arr_all[i:j]
            if len(axis) > 100:
                return axis, len(axis)
    return None, None


# ── Main processing ───────────────────────────────────────────────────────────

def load_maps(folder):
    """Load all .l6m files from folder; return list of (wavenumbers, spectra)."""
    files = sorted(glob.glob(os.path.join(folder, '*.l6m')))
    if not files:
        raise FileNotFoundError(f"No .l6m files found in: {folder}")
    print(f"Found {len(files)} map file(s):")
    for f in files:
        print(f"  {os.path.basename(f)}")

    maps = []
    for fp in files:
        print(f"\nParsing: {os.path.basename(fp)}")
        wn, spec, nx, ny = parse_l6m(fp)
        print(f"  Grid: {nx} x {ny}   |   Wavenumber points: {len(wn)}")
        print(f"  Wavenumber range: {wn[0]:.1f} – {wn[-1]:.1f} cm⁻¹")
        maps.append((wn, spec))
    return maps


def average_maps(maps):
    """
    Average Y-axis spectra point by point across all maps.

    Assumes every map has the same number of Y positions and wavenumber axis.
    Returns (wavenumbers, averaged_spectra) where averaged_spectra[i] is the
    mean spectrum at Y position i across all maps.
    """
    ref_wn = maps[0][0]
    n_y = maps[0][1].shape[0]

    for i, (wn, spec) in enumerate(maps):
        if len(wn) != len(ref_wn):
            raise ValueError(
                f"Map {i+1} has {len(wn)} wavenumber points but map 1 has {len(ref_wn)}. "
                "All maps must share the same spectral axis."
            )
        if spec.shape[0] != n_y:
            raise ValueError(
                f"Map {i+1} has {spec.shape[0]} Y positions but map 1 has {n_y}."
            )

    # Stack along a new axis → shape (n_maps, n_y, n_wn)
    stack = np.stack([m[1] for m in maps], axis=0)
    averaged = stack.mean(axis=0)   # shape (n_y, n_wn)
    return ref_wn, averaged


def plot_results(wavenumbers, averaged_spectra, output_dir=None):
    """Plot all averaged Y-position spectra on one figure."""
    n_y = averaged_spectra.shape[0]
    fig, ax = plt.subplots(figsize=(12, 6))

    cmap = plt.cm.get_cmap('viridis', n_y)
    for i in range(n_y):
        ax.plot(wavenumbers, averaged_spectra[i], color=cmap(i),
                label=f'Y{i+1}', linewidth=0.8)

    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Intensity (counts)', fontsize=12)
    ax.set_title('Point-by-Point Averaged Spectra (all Y positions)', fontsize=13)

    # Colorbar as legend substitute when there are many Y positions
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, n_y))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Y position index')

    plt.tight_layout()

    if output_dir:
        out_path = os.path.join(output_dir, 'averaged_spectra.png')
        plt.savefig(out_path, dpi=150)
        print(f"\nPlot saved to: {out_path}")

    plt.show()


def save_csv(wavenumbers, averaged_spectra, output_dir):
    """Save averaged spectra to CSV: columns = [Wavenumber, Y1, Y2, ..., Y25]."""
    n_y = averaged_spectra.shape[0]
    header = 'Wavenumber(cm-1),' + ','.join(f'Y{i+1}' for i in range(n_y))
    data = np.column_stack([wavenumbers, averaged_spectra.T])
    out_path = os.path.join(output_dir, 'averaged_spectra.csv')
    np.savetxt(out_path, data, delimiter=',', header=header, comments='')
    print(f"CSV saved to: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ▼ Change this path to your actual folder ▼
    FOLDER = r'C:\Users\Hira Aman\Downloads\15 days old drop'

    print("=" * 60)
    print("LabSpec6 Map Averager")
    print("=" * 60)

    maps = load_maps(FOLDER)

    print("\nAveraging Y-axis spectra point by point across all maps…")
    wavenumbers, averaged = average_maps(maps)
    print(f"Done. Averaged array shape: {averaged.shape}  "
          f"({averaged.shape[0]} Y positions × {averaged.shape[1]} wavenumber points)")

    # Save outputs next to the input folder
    output_dir = FOLDER
    save_csv(wavenumbers, averaged, output_dir)
    plot_results(wavenumbers, averaged, output_dir)
