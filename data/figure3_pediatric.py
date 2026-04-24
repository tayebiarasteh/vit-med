"""
Created on April 22, 2026.
figure3_pediatric.py

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os


class Figure3:
    OUT_PNG  = './pediatric_results.png'

    DATASET_DISPLAY = {
        'pedi': 'Pedi-CXR', 'vindr': 'VinDr-CXR', 'cxr14': 'ChestX-ray14',
        'padchest': 'PadChest', 'chexpert': 'CheXpert', 'mimic': 'MIMIC-CXR',
        'UKA': 'UKA-CXR',
    }
    DATASET_ORDER = ['pedi', 'vindr', 'cxr14', 'padchest',
                     'chexpert', 'mimic', 'UKA']
    ADULT_DATASETS = ['vindr', 'cxr14', 'padchest', 'chexpert', 'mimic', 'UKA']

    INIT_DISPLAY = {'imgnet': 'ImageNet', 'dinov2': 'DINOv2', 'dinov3': 'DINOv3'}

    COLOR_IMGNET = '#525252'
    COLOR_DINOV2 = '#4575B4'
    COLOR_DINOV3 = '#D73027'
    INIT_COLOR   = {'imgnet': COLOR_IMGNET,
                    'dinov2': COLOR_DINOV2,
                    'dinov3': COLOR_DINOV3}

    PEDI_LABEL_DISPLAY = {
        'finding': 'No finding',
        'Pneumonia': 'Pneumonia',
        'Bronchiolitis': 'Bronchiolitis',
    }
    PEDI_LABELS_ORDER = ['finding', 'Pneumonia', 'Bronchiolitis']

    def __init__(self):
        df = pd.read_csv(self.CSV_PATH)
        self.df = df
        self.macro = df[df['label'] == 'macro_avg'].copy()
        self._set_rcparams()
        self.fig = None

    @staticmethod
    def _set_rcparams():
        plt.rcParams.update({
            'font.family': 'DejaVu Sans', 'font.size': 12,
            'axes.labelsize': 13, 'axes.titlesize': 13,
            'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5,
            'legend.fontsize': 11,
            'axes.spines.top': False, 'axes.spines.right': False,
            'axes.linewidth': 1.1,
            'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
            'axes.grid': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        })

    def get_macro(self, dataset, init, backbone, size):
        m = self.macro[
            (self.macro['dataset'] == dataset) &
            (self.macro['initialization'] == init) &
            (self.macro['backbone'] == backbone) &
            (self.macro['image_size'] == size)]
        return m.iloc[0] if len(m) else None

    def get_label_row(self, dataset, init, backbone, size, label):
        m = self.df[
            (self.df['dataset'] == dataset) &
            (self.df['initialization'] == init) &
            (self.df['backbone'] == backbone) &
            (self.df['image_size'] == size) &
            (self.df['label'] == label)]
        return m.iloc[0] if len(m) else None

    @staticmethod
    def add_panel_label(ax, letter, title='', x=-0.14, y=1.05):
        ax.text(x, y, letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='bottom', ha='left')
        if title:
            ax.text(x + 0.08, y, title, transform=ax.transAxes,
                    fontsize=14, fontweight='normal', va='bottom', ha='left')

    def build(self):
        self.fig = plt.figure(figsize=(17, 12.5))

        gs_leg = self.fig.add_gridspec(1, 1, left=0.04, right=0.99,
                                       top=0.997, bottom=0.960)
        gs     = self.fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34,
                                       left=0.07, right=0.98,
                                       top=0.920, bottom=0.06)

        self._draw_legend(self.fig.add_subplot(gs_leg[0, 0]))
        self._panel_a(self.fig.add_subplot(gs[0, 0]))
        self._panel_b(self.fig.add_subplot(gs[0, 1]))
        self._panel_c(self.fig.add_subplot(gs[0, 2]))
        self._panel_d(self.fig.add_subplot(gs[1, 0]))
        self._panel_e(self.fig.add_subplot(gs[1, 1]))
        self._panel_f(self.fig.add_subplot(gs[1, 2]))
        return self

    def _draw_legend(self, ax):
        ax.axis('off')
        items = [
            Line2D([0], [0], marker='o', color=self.COLOR_IMGNET,
                   mfc=self.COLOR_IMGNET, ms=8, lw=2.2, label='ImageNet'),
            Line2D([0], [0], marker='o', color=self.COLOR_DINOV2,
                   mfc=self.COLOR_DINOV2, ms=8, lw=2.2, label='DINOv2'),
            Line2D([0], [0], marker='o', color=self.COLOR_DINOV3,
                   mfc=self.COLOR_DINOV3, ms=8, lw=2.2, label='DINOv3'),
            Line2D([0], [0], lw=0, label='   '),
            Line2D([0], [0], marker='s', color='#808080',
                   mec='#808080', ms=9, lw=0, label='ConvNeXt-B'),
            Line2D([0], [0], marker='D', color='#808080',
                   mec='#808080', ms=8, lw=0, label='ViT-B/16'),
            Line2D([0], [0], lw=0, label='   '),
            Line2D([0], [0], marker='o', color='#555', mfc='white',
                   mec='#555', mew=1.6, ms=8, lw=0, label='224 px'),
            Line2D([0], [0], marker='o', color='#555', mfc='#555',
                   ms=8, lw=0, label='512 px'),
            Line2D([0], [0], lw=0, label='   '),
            Patch(fc='#D73027', ec='black', lw=0.5, alpha=0.9,
                  label='Pedi-CXR'),
            Patch(fc='#bdbdbd', ec='black', lw=0.5,
                  label='Adult'),
        ]
        ax.legend(handles=items, loc='center', ncol=12, frameon=False,
                  fontsize=12, handletextpad=0.4, columnspacing=1.0)

    def _panel_a(self, ax):
        """Strip plot of ΔAUROC(512-224) for every config, separated by
        Adult (6 ds pooled) vs Pedi-CXR."""
        configs = [('vitb', 'imgnet'), ('vitb', 'dinov2'), ('vitb', 'dinov3'),
                   ('convnext', 'imgnet'), ('convnext', 'dinov3')]

        adult_deltas, adult_colors = [], []
        pedi_deltas, pedi_colors = [], []

        for ds in self.DATASET_ORDER:
            for bb, init in configs:
                r224 = self.get_macro(ds, init, bb, 224)
                r512 = self.get_macro(ds, init, bb, 512)
                if r224 is None or r512 is None:
                    continue
                delta = r512['AUROC_mean'] - r224['AUROC_mean']
                col = self.INIT_COLOR[init]
                if ds == 'pedi':
                    pedi_deltas.append(delta)
                    pedi_colors.append(col)
                else:
                    adult_deltas.append(delta)
                    adult_colors.append(col)

        rng = np.random.default_rng(42)
        # Adult at x=0, Pedi at x=1
        jit_a = rng.uniform(-0.25, 0.25, size=len(adult_deltas))
        jit_p = rng.uniform(-0.15, 0.15, size=len(pedi_deltas))

        for d, j, c in zip(adult_deltas, jit_a, adult_colors):
            ax.scatter(0 + j, d, s=50, facecolors=c, edgecolors='white',
                       linewidths=0.6, alpha=0.75, zorder=2)
        for d, j, c in zip(pedi_deltas, jit_p, pedi_colors):
            ax.scatter(1 + j, d, s=80, facecolors=c, edgecolors='black',
                       linewidths=0.8, zorder=3)

        # Medians
        med_a = np.median(adult_deltas)
        med_p = np.median(pedi_deltas)
        ax.plot([-0.35, 0.35], [med_a, med_a], color='black', lw=2.2, zorder=4)
        ax.plot([0.65, 1.35], [med_p, med_p], color='#D73027', lw=2.2, zorder=4)
        ax.text(0.38, med_a, f'{med_a:+.2f}', ha='left', va='center',
                fontsize=11, fontweight='bold')
        ax.text(1.38, med_p, f'{med_p:+.2f}', ha='left', va='center',
                fontsize=11, fontweight='bold', color='#D73027')

        ax.axhline(0, color='black', lw=0.8, ls='--', zorder=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Adult\n(6 datasets)', 'Pedi-CXR'], fontsize=12)
        ax.set_ylabel(r'$\Delta$AUROC: 512 $-$ 224 (pp)')
        ax.set_xlim(-0.6, 1.8)
        self.add_panel_label(ax, 'a', 'Resolution lift: adult vs. pediatric')

    def _panel_b(self, ax):
        configs = [('vitb', 'imgnet'), ('vitb', 'dinov2'), ('vitb', 'dinov3'),
                   ('convnext', 'imgnet'), ('convnext', 'dinov3')]

        adult_x, adult_y = [], []
        pedi_x, pedi_y, pedi_c = [], [], []

        for ds in self.DATASET_ORDER:
            for bb, init in configs:
                r224 = self.get_macro(ds, init, bb, 224)
                r512 = self.get_macro(ds, init, bb, 512)
                if r224 is None or r512 is None:
                    continue
                x, y = r224['AUROC_mean'], r512['AUROC_mean']
                if ds == 'pedi':
                    pedi_x.append(x); pedi_y.append(y)
                    pedi_c.append(self.INIT_COLOR[init])
                else:
                    adult_x.append(x); adult_y.append(y)

        ax.scatter(adult_x, adult_y, s=50, facecolors='none',
                   edgecolors='#888', linewidths=1.2, zorder=2)
        for x, y, c in zip(pedi_x, pedi_y, pedi_c):
            ax.scatter(x, y, s=85, facecolors=c, edgecolors='black',
                       linewidths=0.8, zorder=3)

        all_v = np.array(adult_x + adult_y + pedi_x + pedi_y)
        lo = np.floor(all_v.min()) - 0.5
        hi = np.ceil(all_v.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], color='black', lw=1.0, ls='--', zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel('AUROC at 224 (%)')
        ax.set_ylabel('AUROC at 512 (%)')
        ax.set_aspect('equal', adjustable='box')
        self.add_panel_label(ax, 'b', 'Pediatric mostly on diagonal')

    def _panel_c(self, ax):
        configs = [('vitb', 'imgnet'), ('vitb', 'dinov2'), ('vitb', 'dinov3'),
                   ('convnext', 'imgnet'), ('convnext', 'dinov3')]

        datasets = self.DATASET_ORDER
        ranges_list, std_list = [], []
        for ds in datasets:
            vals = []
            for bb, init in configs:
                r = self.get_macro(ds, init, bb, 512)
                if r is not None:
                    vals.append(r['AUROC_mean'])
            ranges_list.append(np.ptp(vals))
            std_list.append(np.std(vals, ddof=1))

        y_pos = np.arange(len(datasets))[::-1]
        colors = ['#D73027' if d == 'pedi' else '#7f7f7f' for d in datasets]

        bars = ax.barh(y_pos, ranges_list, color=colors, edgecolor='black',
                       lw=0.7, alpha=0.85, xerr=std_list, capsize=3,
                       error_kw={'elinewidth': 1.0, 'color': 'black'})

        for y, r, s in zip(y_pos, ranges_list, std_list):
            ax.text(r + s + 0.08, y, f'{r:.2f}', va='center', ha='left',
                    fontsize=11)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.DATASET_DISPLAY[d] for d in datasets])
        ax.set_xlabel('AUROC range across initializations\nat 512 (pp)')
        ax.set_xlim(0, max(r + s for r, s in zip(ranges_list, std_list)) * 1.35)
        self.add_panel_label(ax, 'c', 'Initialization matters least in Pedi-CXR')

    def _panel_d(self, ax):
        configs = [
            ('vitb', 'imgnet', 'ViT-B/16\nImageNet'),
            ('vitb', 'dinov2', 'ViT-B/16\nDINOv2'),
            ('vitb', 'dinov3', 'ViT-B/16\nDINOv3'),
            ('convnext', 'imgnet', 'ConvNeXt-B\nImageNet'),
            ('convnext', 'dinov3', 'ConvNeXt-B\nDINOv3'),
        ]
        x_pos = np.arange(len(configs))

        for i, (bb, init, lbl) in enumerate(configs):
            col = self.INIT_COLOR[init]
            marker = 's' if bb == 'convnext' else 'D'
            r = self.get_macro('pedi', init, bb, 512)
            if r is None:
                continue
            m = r['AUROC_mean']
            lo = r['AUROC_ci_low']
            hi = r['AUROC_ci_high']
            ax.errorbar(x_pos[i], m, yerr=[[m - lo], [hi - m]],
                        fmt=marker, color=col, mec='black', mew=0.5,
                        ms=10, elinewidth=1.8, capsize=5, ecolor=col,
                        zorder=3)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([lbl for _, _, lbl in configs], fontsize=10)
        ax.set_ylabel('AUROC at 512 (%) in Pedi-CXR')
        self.add_panel_label(ax, 'd', 'All Pedi-CXR configurations overlap')

    def _panel_e(self, ax):
        labels = self.PEDI_LABELS_ORDER
        inits = ['imgnet', 'dinov2', 'dinov3']
        backbones = [('convnext', 's'), ('vitb', 'D')]

        n_lbl = len(labels)
        sub_offset_bb = 0.30
        sub_offset_init = 0.09

        x_positions = np.arange(n_lbl).astype(float)

        for li, lbl in enumerate(labels):
            xc = x_positions[li]
            for k_bb, (bb, marker) in enumerate(backbones):
                x_bb = xc + (k_bb - 0.5) * sub_offset_bb
                for k_i, init in enumerate(inits):
                    if init == 'dinov2' and bb == 'convnext':
                        continue
                    x = x_bb + (k_i - 1) * sub_offset_init
                    r224 = self.get_label_row('pedi', init, bb, 224, lbl)
                    r512 = self.get_label_row('pedi', init, bb, 512, lbl)
                    if r224 is None or r512 is None:
                        continue
                    y1 = r224['AUROC_mean']
                    y2 = r512['AUROC_mean']
                    col = self.INIT_COLOR[init]
                    ax.plot([x, x], [y1, y2], color=col, lw=1.8,
                            solid_capstyle='round', zorder=2)
                    ax.plot(x, y1, marker=marker, mfc='white', mec=col,
                            mew=1.4, ms=6.5, zorder=3)
                    ax.plot(x, y2, marker=marker, mfc=col, mec=col,
                            ms=6.5, zorder=3)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([self.PEDI_LABEL_DISPLAY[l] for l in labels],
                           fontsize=11)
        ax.set_ylabel('AUROC (%) in Pedi-CXR')
        self.add_panel_label(ax, 'e', 'Pediatric per-label: no resolution lift')

    def _panel_f(self, ax):
        ds_order = self.DATASET_ORDER

        deltas_per_ds = {}
        for ds in ds_order:
            labs = self.df[(self.df['dataset'] == ds) &
                           (self.df['label'] != 'macro_avg')]['label'].unique()
            vals = []
            for lbl in labs:
                r_ig = self.get_label_row(ds, 'imgnet', 'convnext', 512, lbl)
                r_d3 = self.get_label_row(ds, 'dinov3', 'convnext', 512, lbl)
                if r_ig is not None and r_d3 is not None:
                    vals.append(r_d3['AUROC_mean'] - r_ig['AUROC_mean'])
            deltas_per_ds[ds] = np.array(vals)

        x_pos = np.arange(len(ds_order))
        rng = np.random.default_rng(42)
        for i, ds in enumerate(ds_order):
            vals = deltas_per_ds[ds]
            is_pedi = (ds == 'pedi')
            col = '#D73027' if is_pedi else '#555555'
            face = '#D73027' if is_pedi else '#bdbdbd'

            jx = rng.uniform(-0.20, 0.20, size=len(vals))
            ax.scatter(np.full_like(vals, x_pos[i]) + jx, vals,
                       s=45, facecolors=face, edgecolors=col, lw=0.8,
                       alpha=0.85, zorder=3)

            med = np.median(vals)
            ax.plot([x_pos[i] - 0.30, x_pos[i] + 0.30], [med, med],
                    color=col, lw=2.5, zorder=4)
            ax.text(x_pos[i], med - 0.55 if is_pedi else med + 0.35,
                    f'{med:+.2f}', ha='center',
                    fontsize=10.5, color=col, fontweight='bold')

        ax.axhline(0, color='black', lw=0.8, zorder=1)
        ax.axvspan(-0.5, 0.5, color='#ffe5e5', zorder=0)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([self.DATASET_DISPLAY[d] for d in ds_order],
                           rotation=30, ha='right', fontsize=10.5)
        ax.set_ylabel(r'$\Delta$AUROC: DINOv3 $-$ ImageNet' + '\nat 512 (pp)')
        self.add_panel_label(ax, 'f', 'Per-label DINOv3 gain by cohort')

    def save(self):
        os.makedirs(os.path.dirname(self.OUT_PNG), exist_ok=True)
        self.fig.savefig(self.OUT_PNG, dpi=300, bbox_inches='tight',
                         facecolor='white')


if __name__ == '__main__':
    Figure3().build().save()
