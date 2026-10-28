# -*- coding: utf-8 -*-
"""Talasli Imalat Payi Kontrolu -- Render icin bagimsiz Gradio uygulamasi.
Colab'a ozel hicbir kod icermez (google.colab, files.upload/download yok).
"""
import os

import re
import numpy as np

def parse_iges_scan_curves(path):
    """Parse an IGES file (PC-DMIS style export) and return a dict of
    {curve_name: Nx3 numpy array of XYZ points} for all Type 106 (Copious Data)
    entities that have an associated Type 406 name property."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    dir_lines = [l for l in lines if len(l) > 72 and l[72] == 'D']
    param_lines = [l for l in lines if len(l) > 72 and l[72] == 'P']

    entries = []
    for i in range(0, len(dir_lines), 2):
        e1 = dir_lines[i]
        etype = e1[0:8].strip()
        de_seq = i + 1
        entries.append({'type': etype, 'de_seq': de_seq})

    from collections import defaultdict
    param_by_de = defaultdict(str)
    for l in param_lines:
        back_ptr = l[64:72].strip()
        param_by_de[back_ptr] += l[:64]

    def parse_name(txt):
        m = re.match(r'406,\d+,(\d+)H(.*?);', txt)
        if m:
            n = int(m.group(1))
            return m.group(2)[:n]
        return None

    def parse_106(txt):
        body = txt.split(';')[0]
        parts = body.split(',')
        n = int(parts[2])
        nums = [x for x in parts[3:] if x.strip() != '']
        pts = []
        for k in range(0, min(len(nums), n * 6), 6):
            try:
                x, y, z = float(nums[k]), float(nums[k+1]), float(nums[k+2])
                pts.append((x, y, z))
            except (ValueError, IndexError):
                pass
        return np.array(pts)

    curves = {}
    pending_name = None
    for e in entries:
        txt = param_by_de.get(str(e['de_seq']), '')
        if e['type'] == '406':
            pending_name = parse_name(txt)
        elif e['type'] == '106':
            pts = parse_106(txt)
            if pending_name is not None:
                curves[pending_name] = pts
            pending_name = None
    return curves

import numpy as np

def read_dxf_tags(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read().split('\n')
    raw = [l.rstrip('\r') for l in raw]
    tags = []
    i = 0
    while i < len(raw) - 1:
        code = raw[i].strip()
        val = raw[i+1]
        try:
            code = int(code)
        except ValueError:
            i += 1
            continue
        tags.append((code, val))
        i += 2
    return tags

def _entities_from_tags(tags, section_name):
    in_sec = False
    current = None
    ents = []
    for code, val in tags:
        if code == 2 and val == section_name:
            in_sec = True
            continue
        if code == 0 and val == 'ENDSEC':
            in_sec = False
        if not in_sec:
            continue
        if code == 0:
            if current is not None:
                ents.append(current)
            current = {'type': val, 'codes': []}
        elif current is not None:
            current['codes'].append((code, val))
    if current is not None:
        ents.append(current)
    return ents

def _get_val(codes, c):
    for cc, v in codes:
        if cc == c:
            return float(v)
    return None

def sample_dxf_contour_segments(path, arc_step_deg=2.0):
    """Read a DXF file's model-space geometry and return a LIST of Nx2
    point arrays, one per drawn entity (LINE / ARC / CIRCLE / simple
    INSERT), so each can be drawn as its own connected line instead of
    a single unordered point cloud."""
    tags = read_dxf_tags(path)
    ents = _entities_from_tags(tags, 'ENTITIES')

    block_ents = {}
    in_blocks = False
    block_name = None
    current = None
    for code, val in tags:
        if code == 2 and val == 'BLOCKS':
            in_blocks = True
            continue
        if code == 0 and val == 'ENDSEC':
            in_blocks = False
        if not in_blocks:
            continue
        if code == 0 and val == 'BLOCK':
            current = None
            continue
        if code == 2:
            block_name = val
            block_ents.setdefault(block_name, [])
            continue
        if code == 0:
            if current is not None:
                block_ents[block_name].append(current)
            current = {'type': val, 'codes': []}
            continue
        if current is not None:
            current['codes'].append((code, val))
    if current is not None:
        block_ents[block_name].append(current)

    def entity_to_segment(e):
        if e['type'] == 'LINE':
            x1 = _get_val(e['codes'], 10); y1 = _get_val(e['codes'], 20)
            x2 = _get_val(e['codes'], 11); y2 = _get_val(e['codes'], 21)
            return np.array([[x1, y1], [x2, y2]])
        elif e['type'] == 'ARC':
            cx = _get_val(e['codes'], 10); cy = _get_val(e['codes'], 20)
            r = _get_val(e['codes'], 40)
            a1 = _get_val(e['codes'], 50); a2 = _get_val(e['codes'], 51)
            if a2 < a1:
                a2 += 360
            n = max(2, int((a2 - a1) / arc_step_deg))
            angs = np.radians(a1 + (a2 - a1) * np.arange(n + 1) / n)
            return np.column_stack([cx + r * np.cos(angs), cy + r * np.sin(angs)])
        elif e['type'] == 'CIRCLE':
            cx = _get_val(e['codes'], 10); cy = _get_val(e['codes'], 20)
            r = _get_val(e['codes'], 40)
            n = max(8, int(360 / arc_step_deg))
            angs = np.radians(np.arange(n + 1) * 360 / n)
            return np.column_stack([cx + r * np.cos(angs), cy + r * np.sin(angs)])
        return None

    segments = []
    for e in ents:
        seg = entity_to_segment(e)
        if seg is not None:
            segments.append(seg)
        if e['type'] == 'INSERT':
            bname = ix = iy = None
            ix = iy = 0.0
            for c, v in e['codes']:
                if c == 2:
                    bname = v
                if c == 10:
                    ix = float(v)
                if c == 20:
                    iy = float(v)
            if bname in block_ents:
                for sub_e in block_ents[bname]:
                    sub_seg = entity_to_segment(sub_e)
                    if sub_seg is not None:
                        segments.append(sub_seg + np.array([ix, iy]))
    return segments

def sample_dxf_contour(path, arc_step_deg=2.0):
    """Flat point-cloud version (kept for functions that just need a
    point cloud, e.g. mirror-centre finding)."""
    segs = sample_dxf_contour_segments(path, arc_step_deg)
    return np.vstack(segs) if segs else np.zeros((0, 2))

import numpy as np
from scipy.optimize import minimize
import re

def group_curves_by_scan(curves):
    """Group curve names like 'SCN12 YONU IC 1' by their leading SCNxx tag.
    If no curve matches that pattern, every curve becomes its own group."""
    groups = {}
    for name, pts in curves.items():
        m = re.match(r'(SCN\d+)', name)
        key = m.group(1) if m else name
        groups.setdefault(key, []).append(pts)
    return {k: np.vstack(v) for k, v in groups.items() if len(v) > 0}

def initial_axis_guess(scan_groups):
    """Try to derive an axis purely from the mutual geometry of the scan
    groups (plane-intersection method). Falls back to a global PCA
    direction if there are too few groups / the planes are (near-)
    parallel, so this never crashes even for unusual scan layouts."""
    all_pts = np.vstack(list(scan_groups.values()))
    centroid_all = all_pts.mean(axis=0)
    u, s, vt = np.linalg.svd(all_pts - centroid_all, full_matrices=False)
    pca_dir = vt[0]
    if pca_dir[1] < 0:
        pca_dir = -pca_dir
    pca_guess = (centroid_all, pca_dir)

    if len(scan_groups) < 2:
        return centroid_all, pca_dir

    plane_data = {}
    for name, pts in scan_groups.items():
        centroid = pts.mean(axis=0)
        uu, ss, vvt = np.linalg.svd(pts - centroid, full_matrices=False)
        normal = vvt[-1]
        plane_data[name] = (centroid, normal)

    names = list(plane_data.keys())
    dirs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1 = plane_data[names[i]][1]
            n2 = plane_data[names[j]][1]
            d = np.cross(n1, n2)
            norm = np.linalg.norm(d)
            if norm > 1e-4:            # ignore (near) parallel planes
                d = d / norm
                if d[1] < 0:
                    d = -d
                dirs.append(d)

    if len(dirs) == 0:
        # all groups lie in (near) parallel planes -- plane-intersection
        # cannot resolve the axis; fall back to PCA
        print("Uyari: tarama gruplarinin duzlemleri birbirine cok yakin; "
              "eksen PCA (genel yayilim) yontemiyle tahmin ediliyor.")
        return pca_guess

    d0 = np.mean(dirs, axis=0)
    d0 = d0 / np.linalg.norm(d0)

    A_mat = np.array([n for _, n in plane_data.values()])
    b_vec = np.array([np.dot(n, c) for c, n in plane_data.values()])
    A0, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

    # sanity check: does this axis actually explain the data's own long
    # axis reasonably well? if not, PCA is probably safer.
    if abs(np.dot(d0, pca_dir)) < 0.5:
        print("Uyari: duzlem-kesisim ekseni, verinin genel yayilimiyla "
              "uyusmuyor; PCA yontemine geri donuluyor.")
        return pca_guess

    return A0, d0

def refine_axis_self_consistency(scan_groups, A0, d0, sample_n=1200, trim_frac=0.85):
    if len(scan_groups) < 2:
        # nothing to cross-check against -- keep the initial guess as-is
        return A0, d0

    rng = np.random.RandomState(0)
    samples = {}
    for name, pts in scan_groups.items():
        if len(pts) > sample_n:
            idx = rng.choice(len(pts), sample_n, replace=False)
            samples[name] = pts[idx]
        else:
            samples[name] = pts

    def make_axis(params):
        dtx, dtz, dAx, dAz = params
        d = d0.copy()
        Rx = np.array([[1, 0, 0], [0, np.cos(dtx), -np.sin(dtx)], [0, np.sin(dtx), np.cos(dtx)]])
        Rz = np.array([[np.cos(dtz), -np.sin(dtz), 0], [np.sin(dtz), np.cos(dtz), 0], [0, 0, 1]])
        d = Rz @ (Rx @ d)
        d = d / np.linalg.norm(d)
        A = A0 + np.array([dAx, 0, dAz])
        return A, d

    def project(pts, A, d):
        rel = pts - A
        t = rel @ d
        perp = rel - np.outer(t, d)
        r = np.linalg.norm(perp, axis=1)
        return np.column_stack([t, r])

    from scipy.spatial import cKDTree

    def loss(params):
        A, d = make_axis(params)
        projs = {k: project(v, A, d) for k, v in samples.items()}
        names = list(projs.keys())
        total = 0.0
        base = projs[names[0]]
        tree = cKDTree(base)
        for other in names[1:]:
            dist, _ = tree.query(projs[other])
            n_keep = int(len(dist) * trim_frac)
            total += np.mean(np.sort(dist)[:n_keep] ** 2)
        return total

    res = minimize(loss, np.zeros(4), method='Nelder-Mead',
                    options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 4000, 'maxfev': 4000})
    A_opt, d_opt = make_axis(res.x)
    return A_opt, d_opt

def project_axial_radial(pts, A, d):
    rel = pts - A
    t = rel @ d
    perp = rel - np.outer(t, d)
    r = np.linalg.norm(perp, axis=1)
    return np.column_stack([t, r])

import numpy as np
import re
from scipy.spatial import cKDTree

def build_midline_segments(curves, max_pair_dist=30.0):
    """Like build_midline, but returns a dict {segment_name: Nx3 array}
    so each scan pass can later be drawn as its own connected line.
    - IC/DIS pairs are merged into a true midline where they match closely.
    - Any curve that is NEITHER an IC nor a DIS trace (e.g. 'YAN' / side-
      face passes) is kept as its own raw segment instead of being
      dropped -- earlier versions silently discarded these, which is
      why some flat/face regions were missing from the plot."""
    scn_tags = sorted(set(re.match(r'(SCN\d+)', n).group(1)
                          for n in curves if re.match(r'(SCN\d+)', n)))
    segments = {}
    if not scn_tags:
        segments['ALL'] = np.vstack(list(curves.values()))
        return segments

    for scn in scn_tags:
        ic_keys = [k for k in curves if k.startswith(scn) and 'IC' in k.upper()]
        dis_keys = [k for k in curves if k.startswith(scn) and 'DIS' in k.upper()]
        other_keys = [k for k in curves if k.startswith(scn)
                      and k not in ic_keys and k not in dis_keys]

        ic = np.vstack([curves[k] for k in ic_keys]) if ic_keys else np.zeros((0, 3))
        dis = np.vstack([curves[k] for k in dis_keys]) if dis_keys else np.zeros((0, 3))

        if len(ic) and len(dis):
            tree = cKDTree(dis)
            d, idx = tree.query(ic)
            good = d < max_pair_dist
            mid = ic.copy()
            mid[good] = (ic[good] + dis[idx[good]]) / 2.0
            segments[scn + '_IC'] = mid
            tree_ic = cKDTree(ic)
            d2, idx2 = tree_ic.query(dis)
            unmatched_dis = d2 > max_pair_dist
            if unmatched_dis.any():
                segments[scn + '_DIS_extra'] = dis[unmatched_dis]
        elif len(ic):
            segments[scn + '_IC'] = ic
        elif len(dis):
            segments[scn + '_DIS'] = dis

        # keep every other curve (e.g. 'YAN' side/face passes) as its
        # own segment, in original point order, instead of dropping it
        for k in other_keys:
            segments[k] = curves[k]

    return segments

def build_midline(curves, max_pair_dist=30.0):
    """Flat point-cloud version (kept for axis/alignment functions that
    just need a point cloud, not line connectivity)."""
    segs = build_midline_segments(curves, max_pair_dist)
    return np.vstack(list(segs.values()))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

def _colored_line(ax, pts, colors, cmap='YlOrRd', vmin=0, vmax=15, lw=2.0):
    """Draw a single connected line whose color varies point-to-point."""
    if len(pts) < 2:
        return None
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_colors = (colors[:-1] + colors[1:]) / 2.0
    lc = LineCollection(segs, cmap=cmap, linewidths=lw)
    lc.set_array(seg_colors)
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)
    return lc

def plot_critical_lines(scan_segments, contour_segments, payi_esik_mm,
                         title="Talasli imalat payi"):
    """scan_segments: list of Nx2 arrays (already transformed to the DXF
    plane), one per scan pass/branch.
    contour_segments: list of Nx2 arrays, one per DXF entity."""
    all_contour_pts = np.vstack(contour_segments)
    tree = cKDTree(all_contour_pts)

    fig, ax = plt.subplots(figsize=(9, 9))
    for seg in contour_segments:
        ax.plot(seg[:, 0], seg[:, 1], '-', color='#9ca3af', linewidth=1.2, alpha=0.7)
    ax.plot([], [], '-', color='#9ca3af', linewidth=1.2, label='Islenmis parca')

    all_d = []
    crit_points = []
    lc_ref = None
    for seg in scan_segments:
        d, _ = tree.query(seg)
        all_d.append(d)
        lc_ref = _colored_line(ax, seg, d, vmax=max(15, payi_esik_mm * 5))
        crit_mask = d < payi_esik_mm
        if crit_mask.any():
            crit_points.append(seg[crit_mask])
            ax.plot(seg[crit_mask, 0], seg[crit_mask, 1], '.', color='red', markersize=6)

    if lc_ref is not None:
        cbar = plt.colorbar(lc_ref, ax=ax)
        cbar.set_label('Mesafe (mm)')
    ax.plot([], [], '-', color='red', linewidth=2,
            label=f'Pay < {payi_esik_mm:.1f}mm (kritik)')

    all_d_flat = np.concatenate(all_d) if all_d else np.array([])

    # annotate minimum of each critical cluster (by scan segment, then by
    # contiguous run along that segment)
    for seg in scan_segments:
        d, _ = tree.query(seg)
        idx = np.where(d < payi_esik_mm)[0]
        if len(idx) == 0:
            continue
        clusters, cur = [], [idx[0]]
        for i in idx[1:]:
            if i - cur[-1] > 3:
                clusters.append(cur)
                cur = [i]
            else:
                cur.append(i)
        clusters.append(cur)
        for cl in clusters:
            cl = np.array(cl)
            mi = cl[np.argmin(d[cl])]
            ax.annotate(f'{d[mi]:.2f}mm', (seg[mi, 0], seg[mi, 1]),
                        textcoords="offset points", xytext=(6, 0),
                        fontsize=8, color='darkred', fontweight='bold')

    # DXF ve taramanin Y eksenindeki EN DUSUK (alt uc) noktalari arasindaki
    # fark -- ayri bir bilgi olarak grafige yaziliyor.
    min_y_contour = all_contour_pts[:, 1].min()
    all_scan_pts = np.vstack(scan_segments) if scan_segments else np.zeros((0, 2))
    if len(all_scan_pts):
        min_y_scan = all_scan_pts[:, 1].min()
        alt_uc_fark = min_y_scan - min_y_contour
        ax.text(0.02, 0.02, f'Alt uc farki (tarama - islenmis): {alt_uc_fark:+.2f} mm',
                transform=ax.transAxes, fontsize=9, color='black',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        alt_uc_fark = None

    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    med = np.median(all_d_flat) if len(all_d_flat) else float('nan')
    ax.set_title(f'{title} -- esik {payi_esik_mm:.1f}mm, medyan {med:.2f}mm')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, all_d_flat, ax

import numpy as np

def export_result_dxf_lines(path, scan_segments, contour_segments, payi_esik_mm, extra_texts=None):
    """Write a genuinely R12 (AC1009) compatible DXF -- uses classic
    POLYLINE/VERTEX/SEQEND. This is the most broadly compatible
    combination (tested working in AutoCAD 2026)."""
    from scipy.spatial import cKDTree
    all_contour_pts = np.vstack(contour_segments)
    tree = cKDTree(all_contour_pts)

    lines = []
    lines.append("0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1009\n0\nENDSEC")
    lines.append("0\nSECTION\n2\nTABLES\n0\nENDSEC")
    lines.append("0\nSECTION\n2\nBLOCKS\n0\nENDSEC")

    ent = ["0\nSECTION\n2\nENTITIES"]

    def add_polyline(pts, color, layer):
        if len(pts) < 2:
            return
        ent.append(f"0\nPOLYLINE\n8\n{layer}\n62\n{color}\n66\n1\n70\n0")
        for x, y in pts:
            ent.append(f"0\nVERTEX\n8\n{layer}\n10\n{x:.4f}\n20\n{y:.4f}\n30\n0.0")
        ent.append("0\nSEQEND")

    def add_text(x, y, txt, color, layer, height=4.0):
        ent.append(f"0\nTEXT\n8\n{layer}\n62\n{color}\n10\n{x:.4f}\n20\n{y:.4f}\n30\n0.0\n40\n{height}\n1\n{txt}")

    for seg in contour_segments:
        add_polyline(seg, 8, "ISLENMIS_PARCA")

    for seg in scan_segments:
        d, _ = tree.query(seg)
        ok = d >= payi_esik_mm

        def runs(mask):
            idx = np.where(mask)[0]
            if len(idx) == 0:
                return
            cur = [idx[0]]
            for i in idx[1:]:
                if i - cur[-1] == 1:
                    cur.append(i)
                else:
                    yield cur
                    cur = [i]
            yield cur

        for run in runs(ok):
            add_polyline(seg[run], 2, "PAY_YETERLI")
        for run in runs(~ok):
            add_polyline(seg[run], 1, "PAY_KRITIK")

        idx = np.where(d < payi_esik_mm)[0]
        if len(idx) > 0:
            cur = [idx[0]]
            clusters = []
            for i in idx[1:]:
                if i - cur[-1] > 3:
                    clusters.append(cur)
                    cur = [i]
                else:
                    cur.append(i)
            clusters.append(cur)
            for cl in clusters:
                cl = np.array(cl)
                mi = cl[np.argmin(d[cl])]
                add_text(seg[mi, 0] + 5, seg[mi, 1], f"{d[mi]:.2f}mm", 1, "PAY_ETIKET", 5.0)

    if extra_texts:
        # alt uc farki gibi ek bilgi metinleri (ayri katman)
        for x, y, txt in extra_texts:
            add_text(x, y, txt, 3, "OLCUM_BILGISI", 6.0)

    ent.append("0\nENDSEC")
    lines.append("\n".join(ent))
    lines.append("0\nEOF")

    with open(path, 'w') as f:
        f.write("\n".join(lines))
    return path

def _build_combined(scn3_state, scn9_state, scn12_state, payi_esik_mm, common_dy, contour_segments):
    tree = cKDTree(np.vstack(contour_segments))

    def with_common_y(state, take_slice):
        # o taramanin kendi paneldeki dy'sini cikarip, ortak dy'yi uygular
        # (X ve R, panelde ayarlanan degerlerle AYNEN korunur)
        old_dy = state['dy']
        segs = [seg + np.array([0.0, common_dy - old_dy]) for seg in state['segments'][take_slice]]
        return segs

    scn3_left = with_common_y(scn3_state, slice(0, None, 2))
    scn9_right = with_common_y(scn9_state, slice(1, None, 2))
    scn12_right = with_common_y(scn12_state, slice(1, None, 2))

    fig, ax = plt.subplots(figsize=(9, 9))
    for seg in contour_segments:
        ax.plot(seg[:, 0], seg[:, 1], '-', color='#9ca3af', linewidth=1.2, alpha=0.7)
    ax.plot([], [], '-', color='#9ca3af', linewidth=1.2, label='Islenmis parca')

    all_d = []
    lc_ref = None
    for seg in scn3_left + scn9_right + scn12_right:
        d, _ = tree.query(seg)
        all_d.append(d)
        lc_ref = _colored_line(ax, seg, d, vmax=max(15, payi_esik_mm * 5))

    if lc_ref is not None:
        cbar = plt.colorbar(lc_ref, ax=ax)
        cbar.set_label('Mesafe (mm)')

    for seg in scn3_left:
        d, _ = tree.query(seg)
        crit_mask = d < payi_esik_mm
        if crit_mask.any():
            ax.plot(seg[crit_mask, 0], seg[crit_mask, 1], '.', color='red', markersize=6)
        idx = np.where(crit_mask)[0]
        if len(idx) == 0:
            continue
        clusters, cur = [], [idx[0]]
        for i in idx[1:]:
            if i - cur[-1] > 3:
                clusters.append(cur); cur = [i]
            else:
                cur.append(i)
        clusters.append(cur)
        for cl in clusters:
            cl = np.array(cl)
            mi = cl[np.argmin(d[cl])]
            ax.annotate(f'{d[mi]:.2f}mm', (seg[mi, 0], seg[mi, 1]),
                        textcoords="offset points", xytext=(-30, 0),
                        fontsize=8, color='darkred', fontweight='bold')

    right_pts, right_d = [], []
    for seg in scn9_right + scn12_right:
        d, _ = tree.query(seg)
        crit_mask = d < payi_esik_mm
        if crit_mask.any():
            ax.plot(seg[crit_mask, 0], seg[crit_mask, 1], '.', color='red', markersize=6)
        idx = np.where(crit_mask)[0]
        for i in idx:
            right_pts.append(seg[i])
            right_d.append(d[i])

    if right_pts:
        right_pts = np.array(right_pts)
        right_d = np.array(right_d)
        order = np.argsort(right_pts[:, 1])
        right_pts, right_d = right_pts[order], right_d[order]
        clusters, cur = [], [0]
        for i in range(1, len(right_pts)):
            if right_pts[i, 1] - right_pts[cur[-1], 1] > 3:
                clusters.append(cur); cur = [i]
            else:
                cur.append(i)
        clusters.append(cur)
        for cl in clusters:
            cl = np.array(cl)
            mi = cl[np.argmin(right_d[cl])]
            ax.annotate(f'{right_d[mi]:.2f}mm', (right_pts[mi, 0], right_pts[mi, 1]),
                        textcoords="offset points", xytext=(8, 0),
                        fontsize=8, color='darkred', fontweight='bold')

    ax.plot([], [], '-', color='red', linewidth=2,
            label=f'Pay < {payi_esik_mm:.1f}mm (kritik)')

    all_d_flat = np.concatenate(all_d) if all_d else np.array([])
    min_y_contour = np.vstack(contour_segments)[:, 1].min()

    alt_uc = {}
    for name, segs in [('SCN3', scn3_left), ('SCN9', scn9_right), ('SCN12', scn12_right)]:
        min_y_s = np.vstack(segs)[:, 1].min()
        alt_uc[name] = min_y_s - min_y_contour

    info_text = '\n'.join(f'{k} alt uc farki: {v:+.2f} mm' for k, v in alt_uc.items())
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9, color='black',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    med = np.median(all_d_flat) if len(all_d_flat) else float('nan')
    ax.set_title(f'Birlestirilmis: SCN3 (sol) / SCN9+SCN12 (sag) -- medyan {med:.2f}mm')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, all_d_flat, scn3_left + scn9_right + scn12_right, alt_uc

PAYI_ESIK_MM = 3.0

import gradio as gr

# Gradio arayuzunun kendi ici durumu (pipeline calisinca doldurulur)
G = {}

CIRCLE_CENTER_POINT_G = np.array([1.18951836e-11, -1.24282270e+01, -1.58220266e-10])
MARKED_CENTER_POINT_G = np.array([0.0099490363, 0.0, 1.6565055471])


def gr_run_pipeline(dxf_file, igs_file, payi_default):
    if dxf_file is None or igs_file is None:
        return "Lutfen hem DXF hem IGS dosyasini yukleyin."
    dxf_path_g = dxf_file.name
    igs_path_g = igs_file.name

    curves_g = parse_iges_scan_curves(igs_path_g)
    scan_groups_g = group_curves_by_scan(curves_g)
    A0_g, d0_g = initial_axis_guess(scan_groups_g)
    A_opt_g, d_opt_g = refine_axis_self_consistency(scan_groups_g, A0_g, d0_g)

    rel_c = CIRCLE_CENTER_POINT_G - A_opt_g
    t_comp_c = np.dot(rel_c, d_opt_g)
    perp_c = rel_c - t_comp_c * d_opt_g
    A_opt_g = A_opt_g + perp_c

    midline_segments_3d_g = build_midline_segments(curves_g)
    contour_segments_g = sample_dxf_contour_segments(dxf_path_g)

    t_sign_g = -1
    tr_marked = project_axial_radial(MARKED_CENTER_POINT_G.reshape(1, 3), A_opt_g, d_opt_g)
    t_off_g = -(tr_marked[0, 0] * t_sign_g)
    Xc_g = 0.0

    G.clear()
    G.update(dict(curves=curves_g, A_opt=A_opt_g, d_opt=d_opt_g,
                  contour_segments=contour_segments_g, t_sign=t_sign_g,
                  t_off=t_off_g, Xc=Xc_g, states={}, payi_default=payi_default))

    contour_xy = np.vstack(contour_segments_g)
    return (f"Pipeline tamam. {len(curves_g)} tarama egrisi, "
            f"{len(contour_segments_g)} DXF cizgi parcasi okundu.\n"
            f"DXF Y araligi: {contour_xy[:,1].min():.1f} - {contour_xy[:,1].max():.1f}")


def gr_get_scan_base_segments(scn):
    keys = [k for k in G['curves'] if k.startswith(scn)]
    segs = []
    for k in keys:
        pts3d = G['curves'][k]
        if len(pts3d) < 2:
            continue
        tr = project_axial_radial(pts3d, G['A_opt'], G['d_opt'])
        t = tr[:, 0] * G['t_sign'] + G['t_off']
        r = tr[:, 1]
        segs.append((np.column_stack([G['Xc'] - r, t]), -1))
        segs.append((np.column_stack([G['Xc'] + r, t]), +1))
    return segs


def gr_draw_scan(scn, dx, dy, dr, payi):
    if 'curves' not in G:
        return None, "Once 'Pipeline'i Calistir' butonuna basin."
    base = gr_get_scan_base_segments(scn)
    segs = [xy + np.array([dx + sign * dr, dy]) for xy, sign in base]
    fig, d_all, ax = plot_critical_lines(
        segs, G['contour_segments'], payi,
        title=f"{scn} -- talasli imalat payi (x={dx:+.2f}, y={dy:+.2f}, r={dr:+.2f})")
    min_y_c = np.vstack(G['contour_segments'])[:, 1].min()
    min_y_s = np.vstack(segs)[:, 1].min()
    alt_uc = min_y_s - min_y_c
    G['states'][scn] = {'segments': segs, 'd_all': d_all, 'payi': payi,
                         'dx': dx, 'dy': dy, 'dr': dr, 'alt_uc_farki': alt_uc}
    info = (f"Medyan sapma: {np.median(d_all):.2f} mm\n"
             f"{payi}mm esiginin altinda: {int((d_all < payi).sum())} / {len(d_all)} nokta\n"
             f"Alt uc farki (tarama-islenmis): {alt_uc:+.2f} mm")
    return fig, info


def gr_download_scan(scn):
    st = G.get('states', {}).get(scn)
    if st is None:
        return None
    out_path = f"/tmp/{scn}_talasli_imalat_payi.dxf"
    all_pts = np.vstack(st['segments'])
    min_i = np.argmin(all_pts[:, 1])
    extra_texts = [(all_pts[min_i, 0] + 8, all_pts[min_i, 1],
                     f"Alt uc farki: {st['alt_uc_farki']:+.2f}mm")]
    export_result_dxf_lines(out_path, st['segments'], G['contour_segments'],
                              st['payi'], extra_texts=extra_texts)
    return out_path


def gr_draw_combined(common_dy, payi):
    missing = [s for s in ['SCN3', 'SCN9', 'SCN12'] if s not in G.get('states', {})]
    if missing:
        return None, f"Once su panelleri bir kez cizin: {', '.join(missing)}"
    fig, d_all, segs, alt_uc = _build_combined(
        G['states']['SCN3'], G['states']['SCN9'], G['states']['SCN12'], payi, common_dy,
        G['contour_segments'])
    G['combined'] = {'segments': segs, 'd_all': d_all, 'payi': payi, 'alt_uc': alt_uc}
    info = (f"Medyan sapma: {np.median(d_all):.2f} mm\n"
             f"{payi}mm esiginin altinda: {int((d_all < payi).sum())} / {len(d_all)} nokta\n" +
             "\n".join(f"{k} alt uc farki: {v:+.2f} mm" for k, v in alt_uc.items()))
    return fig, info


def gr_download_combined():
    st = G.get('combined')
    if st is None:
        return None
    out_path = "/tmp/SCN3_SCN9_SCN12_birlestirilmis.dxf"
    all_pts = np.vstack(st['segments'])
    x_span = all_pts[:, 0].max() - all_pts[:, 0].min()
    extra_texts = []
    for i, (k, v) in enumerate(st['alt_uc'].items()):
        px = all_pts[:, 0].min() + x_span * 0.1 + i * x_span * 0.3
        py = all_pts[:, 1].min() - 10
        extra_texts.append((px, py, f"{k} alt uc farki: {v:+.2f}mm"))
    export_result_dxf_lines(out_path, st['segments'], G['contour_segments'],
                              st['payi'], extra_texts=extra_texts)
    return out_path


def _make_scan_tab(scn):
    with gr.Tab(scn):
        with gr.Row():
            dx = gr.Number(value=0.0, label="X kaydirma (mm)")
            dy = gr.Number(value=-3.0, label="Y kaydirma (mm)")
            dr = gr.Number(value=0.0, label="R kaydirma (yaricap, mm)")
            payi = gr.Number(value=PAYI_ESIK_MM, label="Payi esigi (mm)")
        draw_btn = gr.Button("Ciz / Guncelle", variant="primary")
        plot_out = gr.Plot()
        info_out = gr.Textbox(label="Sonuc", interactive=False, lines=3)
        dl_btn = gr.Button("DXF olarak indir")
        dl_file = gr.File(label="Indirilecek dosya")

        draw_btn.click(lambda dx, dy, dr, payi, scn=scn: gr_draw_scan(scn, dx, dy, dr, payi),
                        inputs=[dx, dy, dr, payi], outputs=[plot_out, info_out])
        dl_btn.click(lambda scn=scn: gr_download_scan(scn), inputs=[], outputs=dl_file)


with gr.Blocks(title="Talasli Imalat Payi Kontrolu") as demo:
    gr.Markdown("# Talasli Imalat Payi Kontrolu")
    with gr.Row():
        dxf_file_in = gr.File(label="Islenmis parca DXF dosyasi")
        igs_file_in = gr.File(label="Tarama IGS dosyasi")
        payi0 = gr.Number(value=PAYI_ESIK_MM, label="Varsayilan payi esigi (mm)")
    run_btn = gr.Button("Pipeline'i Calistir", variant="primary")
    status = gr.Textbox(label="Durum", interactive=False, lines=3)
    run_btn.click(gr_run_pipeline, inputs=[dxf_file_in, igs_file_in, payi0], outputs=status)

    with gr.Tabs():
        _make_scan_tab('SCN12')
        _make_scan_tab('SCN3')
        _make_scan_tab('SCN9')

        with gr.Tab("Birlestirilmis"):
            with gr.Row():
                common_dy = gr.Number(value=-3.0, label="Y kaydirma (mm, hepsi)")
                payi_c = gr.Number(value=PAYI_ESIK_MM, label="Payi esigi (mm)")
            draw_btn_c = gr.Button("Guncelle", variant="primary")
            plot_out_c = gr.Plot()
            info_out_c = gr.Textbox(label="Sonuc", interactive=False, lines=4)
            dl_btn_c = gr.Button("DXF olarak indir")
            dl_file_c = gr.File(label="Indirilecek dosya")

            draw_btn_c.click(gr_draw_combined, inputs=[common_dy, payi_c],
                              outputs=[plot_out_c, info_out_c])
            dl_btn_c.click(gr_download_combined, inputs=[], outputs=dl_file_c)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    # Giris sifresi: Render'da Environment sekmesinden APP_USERNAME ve
    # APP_PASSWORD degiskenlerini tanimlayin. Tanimlanmazsa asagidaki
    # varsayilanlar kullanilir -- guvenlik icin mutlaka degistirin.
    app_user = os.environ.get('APP_USERNAME', 'admin')
    app_pass = os.environ.get('APP_PASSWORD', 'degistir123')
    demo.launch(server_name='0.0.0.0', server_port=port,
                auth=(app_user, app_pass))
