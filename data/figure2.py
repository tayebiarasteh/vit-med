"""
Created on April 22, 2026.
figure2.py

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os

df = pd.read_csv(CSV_PATH)
macro = df[df['label'] == 'macro_avg'].copy()

DATASET_DISPLAY = {
    'pedi': 'Pedi-CXR', 'vindr': 'VinDr-CXR', 'cxr14': 'ChestX-ray14',
    'padchest': 'PadChest', 'chexpert': 'CheXpert', 'mimic': 'MIMIC-CXR',
    'UKA': 'UKA-CXR',
}
DATASET_ORDER = ['pedi', 'vindr', 'cxr14', 'padchest', 'chexpert', 'mimic', 'UKA']

INIT_DISPLAY = {'imgnet': 'ImageNet', 'dinov2': 'DINOv2', 'dinov3': 'DINOv3'}
BACK_DISPLAY = {'vitb': 'ViT-B/16', 'convnext': 'ConvNeXt-B'}

COLOR_IMGNET = '#525252'
COLOR_DINOV2 = '#4575B4'
COLOR_DINOV3 = '#D73027'
INIT_COLOR = {'imgnet': COLOR_IMGNET, 'dinov2': COLOR_DINOV2, 'dinov3': COLOR_DINOV3}

HEATMAP_CONFIGS = [
    ('vitb', 'imgnet'), ('vitb', 'dinov2'), ('vitb', 'dinov3'),
    ('convnext', 'imgnet'), ('convnext', 'dinov3'),
]
CONFIG_LABELS = [f'{BACK_DISPLAY[b]}\n{INIT_DISPLAY[i]}' for b, i in HEATMAP_CONFIGS]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10.5,
    'legend.fontsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 1.1, 'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'axes.grid': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

def get_row(dataset, init, backbone, size):
    m = macro[(macro['dataset'] == dataset) & (macro['initialization'] == init) &
              (macro['backbone'] == backbone) & (macro['image_size'] == size)]
    return m.iloc[0] if len(m) else None

def ci_overlap(lo1, hi1, lo2, hi2):
    return not (hi1 < lo2 or hi2 < lo1)

def add_panel_label(ax, letter, title='', x=-0.14, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='bottom', ha='left')
    if title:
        ax.text(x + 0.065, y, title, transform=ax.transAxes,
                fontsize=13, fontweight='normal', va='bottom', ha='left')

def build_auroc_matrix(size):
    mat = np.full((len(DATASET_ORDER), len(HEATMAP_CONFIGS)), np.nan)
    lo  = np.full_like(mat, np.nan)
    hi  = np.full_like(mat, np.nan)
    for i, ds in enumerate(DATASET_ORDER):
        for j, (bb, init) in enumerate(HEATMAP_CONFIGS):
            r = get_row(ds, init, bb, size)
            if r is not None:
                mat[i, j] = r['AUROC_mean']
                lo[i, j]  = r['AUROC_ci_low']
                hi[i, j]  = r['AUROC_ci_high']
    return mat, lo, hi

AUROC_224, CI_LO_224, CI_HI_224 = build_auroc_matrix(224)
AUROC_512, CI_LO_512, CI_HI_512 = build_auroc_matrix(512)
vmin = np.nanmin([AUROC_224, AUROC_512])
vmax = np.nanmax([AUROC_224, AUROC_512])

fig = plt.figure(figsize=(17, 17))

gs_legend = fig.add_gridspec(1, 1, left=0.04, right=0.99, top=0.995, bottom=0.925)
gs = fig.add_gridspec(3, 3, hspace=0.52, wspace=0.36,
                      left=0.08, right=0.97, top=0.880, bottom=0.08)

ax_leg = fig.add_subplot(gs_legend[0, 0])
ax_leg.axis('off')

leg_items = [
    Line2D([0], [0], marker='o', color=COLOR_IMGNET, mfc=COLOR_IMGNET, ms=8, lw=2, label='ImageNet'),
    Line2D([0], [0], marker='o', color=COLOR_DINOV2, mfc=COLOR_DINOV2, ms=8, lw=2, label='DINOv2'),
    Line2D([0], [0], marker='o', color=COLOR_DINOV3, mfc=COLOR_DINOV3, ms=8, lw=2, label='DINOv3'),
    Line2D([0], [0], lw=0, label='   '),
    Line2D([0], [0], marker='o', color='#555', mfc='white', mec='#555', mew=1.6, ms=8, lw=0, label='224 px'),
    Line2D([0], [0], marker='o', color='#555', mfc='#555', ms=8, lw=0, label='512 px'),
    Line2D([0], [0], lw=0, label='   '),
    Line2D([0], [0], marker='s', color='#808080', mec='k', mew=0.5, ms=9, lw=0, label='ConvNeXt-B'),
    Line2D([0], [0], marker='D', color='#808080', mec='k', mew=0.5, ms=8, lw=0, label='ViT-B/16'),
]
ax_leg.legend(handles=leg_items, loc='upper center', bbox_to_anchor=(0.5, 0.98),
              ncol=9, frameon=False, fontsize=10.5,
              handletextpad=0.4, columnspacing=1.2)


def draw_heatmap(ax, mat, letter, title):
    cmap = plt.get_cmap('viridis')
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, '-', ha='center', va='center', color='k', fontsize=10)
                continue
            nv = (v - vmin) / (vmax - vmin + 1e-9)
            tc = 'white' if nv < 0.55 else 'black'
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', color=tc, fontsize=10)
    ax.set_xticks(range(len(HEATMAP_CONFIGS)))
    ax.set_xticklabels(CONFIG_LABELS, rotation=35, ha='right', fontsize=9.5)
    ax.set_yticks(range(len(DATASET_ORDER)))
    ax.set_yticklabels([DATASET_DISPLAY[d] for d in DATASET_ORDER], fontsize=10.5)
    ax.axhline(0.5, color='white', lw=2.2)
    ax.tick_params(axis='both', which='both', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    add_panel_label(ax, letter, title)
    return im

ax_a = fig.add_subplot(gs[0, 0])
im_a = draw_heatmap(ax_a, AUROC_224, 'a', 'AUROC at 224 \u00d7 224')

ax_b = fig.add_subplot(gs[0, 1])
im_b = draw_heatmap(ax_b, AUROC_512, 'b', 'AUROC at 512 \u00d7 512')

ax_c = fig.add_subplot(gs[0, 2])
DELTA = AUROC_512 - AUROC_224
sig_c = np.zeros_like(DELTA, dtype=bool)
for i in range(DELTA.shape[0]):
    for j in range(DELTA.shape[1]):
        if np.isnan(DELTA[i, j]):
            continue
        sig_c[i, j] = not ci_overlap(CI_LO_224[i, j], CI_HI_224[i, j],
                                     CI_LO_512[i, j], CI_HI_512[i, j])

abs_max = np.nanmax(np.abs(DELTA))
norm_c = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
cmap_div = plt.get_cmap('RdBu_r')
im_c = ax_c.imshow(DELTA, cmap=cmap_div, norm=norm_c, aspect='auto')

for i in range(DELTA.shape[0]):
    for j in range(DELTA.shape[1]):
        v = DELTA[i, j]
        if np.isnan(v):
            ax_c.text(j, i, '-', ha='center', va='center', color='k', fontsize=10)
            continue
        nv = v / abs_max
        tc = 'white' if abs(nv) > 0.6 else 'black'
        star = '*' if sig_c[i, j] else ''
        ax_c.text(j, i, f'{v:+.1f}{star}', ha='center', va='center',
                  color=tc, fontsize=9.5)

ax_c.set_xticks(range(len(HEATMAP_CONFIGS)))
ax_c.set_xticklabels(CONFIG_LABELS, rotation=35, ha='right', fontsize=9.5)
ax_c.set_yticks(range(len(DATASET_ORDER)))
ax_c.set_yticklabels([DATASET_DISPLAY[d] for d in DATASET_ORDER], fontsize=10.5)
ax_c.axhline(0.5, color='white', lw=2.2)
ax_c.tick_params(axis='both', which='both', length=0)
for s in ax_c.spines.values():
    s.set_visible(False)
add_panel_label(ax_c, 'c', r'$\Delta$AUROC (512 $-$ 224)')

pos_leg = ax_leg.get_position()
cbar_row_y = pos_leg.y0 + 0.18 * pos_leg.height
cbar_h = 0.12 * pos_leg.height
cbar_w = 0.22 * pos_leg.width
cbar_gap = 0.08 * pos_leg.width
center_x = pos_leg.x0 + 0.5 * pos_leg.width

cbar_ax_a = fig.add_axes([
    center_x - 0.5 * cbar_gap - cbar_w,
    cbar_row_y,
    cbar_w,
    cbar_h
])
cb = fig.colorbar(im_a, cax=cbar_ax_a, orientation='horizontal')
cb.set_label('AUROC (%)', fontsize=10, labelpad=2)
cb.ax.tick_params(labelsize=9)

cbar_ax_c = fig.add_axes([
    center_x + 0.5 * cbar_gap,
    cbar_row_y,
    cbar_w,
    cbar_h
])
cb_c = fig.colorbar(im_c, cax=cbar_ax_c, orientation='horizontal')
cb_c.set_label(r'$\Delta$AUROC (pp)', fontsize=10, labelpad=2)
cb_c.ax.tick_params(labelsize=9)


def draw_dumbbells(ax, backbone, inits, letter, title):
    n_ds = len(DATASET_ORDER)
    n_init = len(inits)
    sub_off = 0.22
    y_positions = np.arange(n_ds)[::-1].astype(float)

    for i, ds in enumerate(DATASET_ORDER):
        yc = y_positions[i]
        for k, init in enumerate(inits):
            y = yc + (k - (n_init - 1) / 2) * sub_off
            r224 = get_row(ds, init, backbone, 224)
            r512 = get_row(ds, init, backbone, 512)
            if r224 is None or r512 is None:
                continue
            x1, x2 = r224['AUROC_mean'], r512['AUROC_mean']
            col = INIT_COLOR[init]
            ax.plot([x1, x2], [y, y], color=col, lw=2.0, solid_capstyle='round', zorder=2)
            ax.plot(x1, y, 'o', mfc='white', mec=col, mew=1.8, ms=6, zorder=3)
            ax.plot(x2, y, 'o', mfc=col, mec=col, ms=6, zorder=3)

    sep_y = y_positions[0] - 0.5
    ax.axhline(sep_y, color='lightgrey', lw=1.0, ls='--')
    ax.set_yticks(y_positions)
    ax.set_yticklabels([DATASET_DISPLAY[d] for d in DATASET_ORDER], fontsize=10)
    ax.set_xlabel('AUROC (%)', fontsize=11)
    add_panel_label(ax, letter, title)

ax_d = fig.add_subplot(gs[1, 0])
draw_dumbbells(ax_d, 'vitb', ['imgnet', 'dinov2', 'dinov3'], 'd', 'ViT-B/16: 224 to 512')

ax_f = fig.add_subplot(gs[1, 2])
draw_dumbbells(ax_f, 'convnext', ['imgnet', 'dinov3'], 'f', 'ConvNeXt-B: 224 to 512')

ax_e = fig.add_subplot(gs[1, 1])

def delta_ci(ds, init_a, init_b, backbone, size):
    ra = get_row(ds, init_a, backbone, size)
    rb = get_row(ds, init_b, backbone, size)
    if ra is None or rb is None:
        return np.nan, np.nan, np.nan
    d = ra['AUROC_mean'] - rb['AUROC_mean']
    hwa = (ra['AUROC_ci_high'] - ra['AUROC_ci_low']) / 2.0
    hwb = (rb['AUROC_ci_high'] - rb['AUROC_ci_low']) / 2.0
    hw = np.sqrt(hwa**2 + hwb**2)
    return d, d - hw, d + hw

y_pos = np.arange(len(DATASET_ORDER))[::-1].astype(float)
jitter = 0.15
mkr = {'convnext': 's', 'vitb': 'D'}
clr_e = {'convnext': '#D73027', 'vitb': '#F46D43'}

for k, bb in enumerate(['convnext', 'vitb']):
    off = (k - 0.5) * jitter * 2
    for i, ds in enumerate(DATASET_ORDER):
        d, lo, hi = delta_ci(ds, 'dinov3', 'imgnet', bb, 512)
        if np.isnan(d):
            continue
        y = y_pos[i] + off
        ax_e.errorbar(d, y, xerr=[[d - lo], [hi - d]],
                      fmt=mkr[bb], color=clr_e[bb], mec='k', mew=0.5,
                      ms=7, elinewidth=1.3, capsize=3, ecolor=clr_e[bb])

ax_e.axvline(0, color='black', lw=0.7, ls='--', zorder=1)
sep_y = y_pos[0] - 0.5
ax_e.axhline(sep_y, color='lightgrey', lw=1.0, ls='--')
ax_e.set_yticks(y_pos)
ax_e.set_yticklabels([DATASET_DISPLAY[d] for d in DATASET_ORDER], fontsize=10)
ax_e.set_xlabel(r'$\Delta$AUROC (pp): DINOv3 $-$ ImageNet at 512', fontsize=10)
add_panel_label(ax_e, 'e', 'DINOv3 vs ImageNet at 512')


# Load timing data
tdf = pd.read_csv(TIMING_CSV)
tdf.columns = [c.replace(' ', '_') for c in tdf.columns]

BB_TIMING_MAP = {'vitb': 'vit', 'convnext': 'convnext'}

def get_timing(dataset, backbone, size, init):
    bb_t = BB_TIMING_MAP[backbone]
    r = tdf[(tdf['dataset'] == dataset) & (tdf['backbone'] == bb_t) &
            (tdf['image_size'] == size)]
    if not len(r):
        return None
    r = r.iloc[0]
    t = r[f'{init}_epoch_time']
    n = r[f'{init}_epoch_num']
    if pd.isna(t) or pd.isna(n):
        return None
    return float(t), float(n), float(t) * float(n)

ax_g = fig.add_subplot(gs[2, 0])

DS_1024 = ['pedi', 'cxr14', 'mimic']

DELTA_1024 = np.full((len(DS_1024), len(HEATMAP_CONFIGS)), np.nan)
CI_LO_1024_DELTA = np.full_like(DELTA_1024, np.nan)
sig_1024 = np.zeros_like(DELTA_1024, dtype=bool)

for i, ds in enumerate(DS_1024):
    for j, (bb, init) in enumerate(HEATMAP_CONFIGS):
        r512 = get_row(ds, init, bb, 512)
        r1024 = get_row(ds, init, bb, 1024)
        if r512 is None or r1024 is None:
            continue
        DELTA_1024[i, j] = r1024['AUROC_mean'] - r512['AUROC_mean']
        sig_1024[i, j] = not ci_overlap(r512['AUROC_ci_low'], r512['AUROC_ci_high'],
                                        r1024['AUROC_ci_low'], r1024['AUROC_ci_high'])

abs_max_g = np.nanmax(np.abs(DELTA_1024))
norm_g = TwoSlopeNorm(vmin=-abs_max_g, vcenter=0, vmax=abs_max_g)
im_g = ax_g.imshow(DELTA_1024, cmap='RdBu_r', norm=norm_g, aspect='auto')

for i in range(DELTA_1024.shape[0]):
    for j in range(DELTA_1024.shape[1]):
        v = DELTA_1024[i, j]
        if np.isnan(v):
            ax_g.text(j, i, '-', ha='center', va='center', color='k', fontsize=10)
            continue
        nv = v / abs_max_g
        tc = 'white' if abs(nv) > 0.6 else 'black'
        star = '*' if sig_1024[i, j] else ''
        ax_g.text(j, i, f'{v:+.1f}{star}', ha='center', va='center',
                  color=tc, fontsize=10)

ax_g.set_xticks(range(len(HEATMAP_CONFIGS)))
ax_g.set_xticklabels(CONFIG_LABELS, rotation=35, ha='right', fontsize=9.5)
ax_g.set_yticks(range(len(DS_1024)))
ax_g.set_yticklabels([DATASET_DISPLAY[d] for d in DS_1024], fontsize=10.5)
ax_g.tick_params(axis='both', which='both', length=0)
for s in ax_g.spines.values():
    s.set_visible(False)
add_panel_label(ax_g, 'g', r'$\Delta$AUROC (1024 $-$ 512)')

pos_g = ax_g.get_position()
cbar_g_w = 0.78 * pos_g.width
cbar_g_h = 0.008
cbar_g_x = pos_g.x0 + 0.11 * pos_g.width
cbar_g_y = pos_g.y0 - 0.055
cbar_g_ax = fig.add_axes([cbar_g_x, cbar_g_y, cbar_g_w, cbar_g_h])
cb_g = fig.colorbar(im_g, cax=cbar_g_ax, orientation='horizontal')
cb_g.set_label(r'$\Delta$AUROC (pp)', fontsize=9.5, labelpad=2)
cb_g.ax.tick_params(labelsize=9)

ax_h = fig.add_subplot(gs[2, 1])

def collect_ratios(size_from, size_to):
    """Return dict {(backbone, init): [ratios across datasets]}."""
    results = {}
    for _, r_from in tdf[tdf['image_size'] == size_from].iterrows():
        r_to = tdf[(tdf['dataset'] == r_from['dataset']) &
                   (tdf['backbone'] == r_from['backbone']) &
                   (tdf['image_size'] == size_to)]
        if not len(r_to):
            continue
        r_to = r_to.iloc[0]
        for init in ['imgnet', 'dinov2', 'dinov3']:
            c = f'{init}_epoch_time'
            if pd.notna(r_from[c]) and pd.notna(r_to[c]) and r_from[c] > 0:
                key = (r_from['backbone'], init)
                results.setdefault(key, []).append(r_to[c] / r_from[c])
    return results

ratios_512 = collect_ratios(224, 512)
ratios_1024 = collect_ratios(512, 1024)

BARS_ORDER = [
    ('vit', 'imgnet', COLOR_IMGNET, 'ViT/ImgNet'),
    ('vit', 'dinov2', COLOR_DINOV2, 'ViT/DINOv2'),
    ('vit', 'dinov3', COLOR_DINOV3, 'ViT/DINOv3'),
    ('convnext', 'imgnet', COLOR_IMGNET, 'CNX/ImgNet'),
    ('convnext', 'dinov3', COLOR_DINOV3, 'CNX/DINOv3'),
]

n_bars = len(BARS_ORDER)
group_width = 0.75
bar_width = group_width / n_bars

x_groups = np.array([0, 1.2])  # two transitions with gap

bar_meta = {0: {'vit': [], 'convnext': []}, 1: {'vit': [], 'convnext': []}}
all_bar_tops = []

for i, (bb, init, col, lbl) in enumerate(BARS_ORDER):
    means = []
    stds = []
    for ratios in [ratios_512, ratios_1024]:
        vals = ratios.get((bb, init), [])
        if len(vals) >= 1:
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
        else:
            means.append(np.nan)
            stds.append(0)
    offset = (i - (n_bars - 1) / 2) * bar_width
    xs = x_groups + offset
    hatch = '//' if bb == 'convnext' else None
    ax_h.bar(xs, means, bar_width, yerr=stds, color=col,
             edgecolor='black', lw=0.6, hatch=hatch, capsize=2.5,
             error_kw={'elinewidth': 0.9})

    for gi in range(len(x_groups)):
        if not np.isnan(means[gi]):
            bar_meta[gi][bb].append({'x': xs[gi], 'y': means[gi], 'err': stds[gi]})
            all_bar_tops.append(means[gi] + stds[gi])

ax_h.axhline(1.0, color='black', lw=0.7, ls=':', alpha=0.5)

ax_h.set_xticks(x_groups)
ax_h.set_xticklabels([r'224 $\rightarrow$ 512', r'512 $\rightarrow$ 1024'], fontsize=11)
ax_h.set_ylabel('Epoch-time multiplier (x)', fontsize=11)

ymax_h = max(all_bar_tops) if all_bar_tops else 1.0
ax_h.set_ylim(0, ymax_h * 1.18)

for gi, ratios in enumerate([ratios_512, ratios_1024]):
    all_vit = [v for (bb, _), vs in ratios.items() if bb == 'vit' for v in vs]
    all_cnx = [v for (bb, _), vs in ratios.items() if bb == 'convnext' for v in vs]
    y_offset = 0.025 * ax_h.get_ylim()[1]

    if all_vit and len(bar_meta[gi]['vit']) > 0:
        vit_x = np.mean([d['x'] for d in bar_meta[gi]['vit']])
        vit_y = max(d['y'] + d['err'] for d in bar_meta[gi]['vit']) + y_offset
        ax_h.text(vit_x, vit_y,
                  f'ViT: {np.mean(all_vit):.1f}x',
                  ha='center', va='bottom', fontsize=9, color='#333',
                  style='italic')

    if all_cnx and len(bar_meta[gi]['convnext']) > 0:
        cnx_x = np.mean([d['x'] for d in bar_meta[gi]['convnext']])
        cnx_y = max(d['y'] + d['err'] for d in bar_meta[gi]['convnext']) + y_offset
        ax_h.text(cnx_x, cnx_y,
                  f'CNX: {np.mean(all_cnx):.1f}x',
                  ha='center', va='bottom', fontsize=9, color='#333',
                  style='italic')

add_panel_label(ax_h, 'h', 'Compute cost of higher resolution')

ax_i = fig.add_subplot(gs[2, 2])

points = []
for ds in DS_1024:
    for bb, init in HEATMAP_CONFIGS:
        r512 = get_row(ds, init, bb, 512)
        r1024 = get_row(ds, init, bb, 1024)
        t512 = get_timing(ds, bb, 512, init)
        t1024 = get_timing(ds, bb, 1024, init)
        if r512 is None or r1024 is None or t512 is None or t1024 is None:
            continue
        delta_auroc = r1024['AUROC_mean'] - r512['AUROC_mean']
        cost_ratio = t1024[2] / t512[2]
        points.append({
            'dataset': ds, 'backbone': bb, 'init': init,
            'cost_ratio': cost_ratio, 'delta_auroc': delta_auroc,
            'sig': not ci_overlap(r512['AUROC_ci_low'], r512['AUROC_ci_high'],
                                  r1024['AUROC_ci_low'], r1024['AUROC_ci_high']),
        })

for p in points:
    col = INIT_COLOR[p['init']]
    marker = 's' if p['backbone'] == 'convnext' else 'D'
    ec = 'black' if p['sig'] else col
    ew = 1.2 if p['sig'] else 0.5
    ax_i.scatter(p['cost_ratio'], p['delta_auroc'], marker=marker,
                 s=110, c=col, edgecolors=ec, linewidths=ew, zorder=3)

ax_i.axhline(0, color='black', lw=0.7, ls='--', alpha=0.6)
ax_i.axvline(1, color='black', lw=0.7, ls=':', alpha=0.4)
ax_i.axhspan(-5, 0, xmin=0, xmax=1, color='#ffcccc', alpha=0.25, zorder=0)

ax_i.set_xscale('log')
ax_i.set_xlabel('Total wall-clock time ratio (1024 / 512)', fontsize=10.5)
ax_i.set_ylabel(r'$\Delta$AUROC (1024 $-$ 512) (pp)', fontsize=10.5)

n_neg = sum(1 for p in points if p['delta_auroc'] <= 0)
n_sig_pos = sum(1 for p in points if p['delta_auroc'] > 0 and p['sig'])
n_total = len(points)
summary = (f'{n_sig_pos}/{n_total} significant gains\n'
           f'{n_neg}/{n_total} zero or loss')
ax_i.text(0.97, 0.97, summary, transform=ax_i.transAxes,
          ha='right', va='top', fontsize=9, color='#333',
          bbox=dict(boxstyle='round,pad=0.3', fc='white',
                    ec='#ccc', lw=0.6))

add_panel_label(ax_i, 'i', 'Gain per compute cost (1024 vs 512)')

shape_leg = [
    Line2D([0], [0], marker='s', color='#808080', mec='k', mew=0.5, ms=9,
           lw=0, label='ConvNeXt-B'),
    Line2D([0], [0], marker='D', color='#808080', mec='k', mew=0.5, ms=8,
           lw=0, label='ViT-B/16'),
]
ax_i.legend(handles=shape_leg, loc='lower left', fontsize=9,
            frameon=False, handletextpad=0.3)

fig.savefig('./figure2.png', dpi=300, bbox_inches='tight', facecolor='white')
