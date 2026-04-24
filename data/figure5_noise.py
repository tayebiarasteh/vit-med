"""
Created on April 22, 2026.
figure5_noise.py

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os


class Figure5:
    OUT_PNG = './noise_results.png'

    INITS = ['imgnet', 'dinov2', 'dinov3']
    INIT_DISPLAY = {'imgnet': 'ImageNet', 'dinov2': 'DINOv2', 'dinov3': 'DINOv3'}
    NOISE_RATIOS = [0, 10, 20, 30, 40]

    COLOR_IMGNET = '#525252'
    COLOR_DINOV2 = '#4575B4'
    COLOR_DINOV3 = '#D73027'
    INIT_COLOR = {'imgnet': COLOR_IMGNET,
                  'dinov2': COLOR_DINOV2,
                  'dinov3': COLOR_DINOV3}

    LABEL_DISPLAY = {
        'effusion': 'Effusion',
        'fibrosis': 'Fibrosis',
        'Atelectasis': 'Atelectasis',
        'Cardiomegaly': 'Cardiomegaly',
        'Consolidation': 'Consolidation',
        'Mass': 'Mass',
        'Opacity': 'Opacity',
        'Pneumonia': 'Pneumonia',
        'Pneumothorax': 'Pneumothorax',
        'finding': 'No finding',
        'thickening': 'Thickening',
    }

    def __init__(self):
        self.df = pd.read_csv(self.NOISE_CSV)
        self.macro = self.df[self.df['label'] == 'macro_avg'].copy()
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

    def get_macro(self, init, noise):
        m = self.macro[(self.macro['initialization'] == init) &
                       (self.macro['noise_ratio'] == noise)]
        return m.iloc[0] if len(m) else None

    def get_label(self, init, noise, label):
        m = self.df[(self.df['initialization'] == init) &
                    (self.df['noise_ratio'] == noise) &
                    (self.df['label'] == label)]
        return m.iloc[0] if len(m) else None

    @staticmethod
    def ci_overlap(lo1, hi1, lo2, hi2):
        return not (hi1 < lo2 or hi2 < lo1)

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
                                       top=0.997, bottom=0.955)
        gs = self.fig.add_gridspec(2, 3, hspace=0.48, wspace=0.36,
                                   left=0.07, right=0.98,
                                   top=0.915, bottom=0.08)

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
                   mfc=self.COLOR_IMGNET, ms=9, lw=2.4, label='ImageNet'),
            Line2D([0], [0], marker='o', color=self.COLOR_DINOV2,
                   mfc=self.COLOR_DINOV2, ms=9, lw=2.4, label='DINOv2'),
            Line2D([0], [0], marker='o', color=self.COLOR_DINOV3,
                   mfc=self.COLOR_DINOV3, ms=9, lw=2.4, label='DINOv3'),
        ]
        ax.legend(handles=items, loc='center', ncol=3, frameon=False,
                  fontsize=12, handletextpad=0.5, columnspacing=2.0)

    def _panel_a(self, ax):
        for init in self.INITS:
            means, cilo, cihi = [], [], []
            for nr in self.NOISE_RATIOS:
                r = self.get_macro(init, nr)
                if r is not None:
                    means.append(r['AUROC_mean'])
                    cilo.append(r['AUROC_ci_low'])
                    cihi.append(r['AUROC_ci_high'])
            means = np.array(means)
            cilo = np.array(cilo)
            cihi = np.array(cihi)
            col = self.INIT_COLOR[init]
            ax.fill_between(self.NOISE_RATIOS, cilo, cihi, color=col,
                            alpha=0.18, lw=0, zorder=1)
            ax.plot(self.NOISE_RATIOS, means, 'o-', color=col, lw=2.5, ms=8,
                    mec='white', mew=1.0, zorder=3)

        ax.set_xticks(self.NOISE_RATIOS)
        ax.set_xlabel('Label noise ratio (%)')
        ax.set_ylabel('AUROC (%)')
        self.add_panel_label(ax, 'a', 'AUROC degrades with label noise')

    def _panel_b(self, ax):
        for init in self.INITS:
            means, cilo, cihi = [], [], []
            for nr in self.NOISE_RATIOS:
                r = self.get_macro(init, nr)
                if r is not None:
                    means.append(r['mAP_mean'])
                    cilo.append(r['mAP_ci_low'])
                    cihi.append(r['mAP_ci_high'])
            means = np.array(means)
            cilo = np.array(cilo)
            cihi = np.array(cihi)
            col = self.INIT_COLOR[init]
            ax.fill_between(self.NOISE_RATIOS, cilo, cihi, color=col,
                            alpha=0.18, lw=0, zorder=1)
            ax.plot(self.NOISE_RATIOS, means, 'o-', color=col, lw=2.5, ms=8,
                    mec='white', mew=1.0, zorder=3)

        ax.set_xticks(self.NOISE_RATIOS)
        ax.set_xlabel('Label noise ratio (%)')
        ax.set_ylabel('mAP (%)')
        self.add_panel_label(ax, 'b', 'mAP shows the same pattern')

    def _panel_c(self, ax):
        width = 0.27
        x = np.arange(len(self.NOISE_RATIOS) - 1)
        noise_levels = [nr for nr in self.NOISE_RATIOS if nr > 0]

        for i, init in enumerate(self.INITS):
            col = self.INIT_COLOR[init]
            baseline = self.get_macro(init, 0)['AUROC_mean']
            drops = []
            for nr in noise_levels:
                r = self.get_macro(init, nr)
                drops.append(r['AUROC_mean'] - baseline)
            offset = (i - 1) * width
            # Draw bars with negative values
            ax.bar(x + offset, drops, width, color=col, edgecolor='black',
                   lw=0.6, label=self.INIT_DISPLAY[init])

            for xi, d in zip(x, drops):
                ax.text(xi + offset, d - 0.7,
                        f'{d:+.1f}', ha='center', va='top',
                        fontsize=9, color='black', fontweight='bold')

        ax.axhline(0, color='black', lw=0.8, ls='--', zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{nr}%' for nr in noise_levels])
        ax.set_xlabel('Label noise ratio (%)')
        ax.set_ylabel(r'$\Delta$AUROC from 0% noise (pp)')
        ax.set_ylim(-14, 0)
        ax.invert_yaxis()
        self.add_panel_label(ax, 'c', 'DINOv3 loses more AUROC under noise')

    def _panel_d(self, ax):
        d3_ig = []
        d3_d2 = []
        for nr in self.NOISE_RATIOS:
            d3 = self.get_macro('dinov3', nr)['AUROC_mean']
            ig = self.get_macro('imgnet', nr)['AUROC_mean']
            d2 = self.get_macro('dinov2', nr)['AUROC_mean']
            d3_ig.append(d3 - ig)
            d3_d2.append(d3 - d2)

        ax.plot(self.NOISE_RATIOS, d3_ig, 'o-', color=self.COLOR_IMGNET,
                lw=2.4, ms=9, mec='white', mew=1.0, zorder=3,
                label='DINOv3 minus ImageNet')
        ax.plot(self.NOISE_RATIOS, d3_d2, 's-', color=self.COLOR_DINOV2,
                lw=2.4, ms=9, mec='white', mew=1.0, zorder=3,
                label='DINOv3 minus DINOv2')

        ax.axhline(0, color='black', lw=0.8, ls='--', zorder=1)

        ax.axhspan(-4, 0, color='#ffe5e5', alpha=0.35, zorder=0)

        ax.set_xticks(self.NOISE_RATIOS)
        ax.set_xlabel('Label noise ratio (%)')
        ax.set_ylabel(r'AUROC advantage of DINOv3 (pp)')
        ax.set_ylim(-4, None)  # Start from -4
        self.add_panel_label(ax, 'd', 'DINOv3 advantage erodes and reverses')

    def _panel_e(self, ax):
        """Rows = labels, cols = inits, cell = AUROC drop from 0% to 40%."""
        labs = sorted(set(self.df[self.df['label'] != 'macro_avg']
                          ['label'].unique()),
                      key=lambda x: -self._label_mean_drop(x))

        data = np.zeros((len(labs), len(self.INITS)))
        for i, lbl in enumerate(labs):
            for j, init in enumerate(self.INITS):
                r0 = self.get_label(init, 0, lbl)
                r40 = self.get_label(init, 40, lbl)
                if r0 is not None and r40 is not None:
                    data[i, j] = r0['AUROC_mean'] - r40['AUROC_mean']

        vmax = np.max(np.abs(data))
        im = ax.imshow(data, cmap='Reds', vmin=0, vmax=vmax, aspect='auto')
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                txt_color = 'white' if v > vmax * 0.55 else 'black'
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                        fontsize=10, color=txt_color, fontweight='bold')

        ax.set_xticks(range(len(self.INITS)))
        ax.set_xticklabels([self.INIT_DISPLAY[i] for i in self.INITS],
                           fontsize=10.5)
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels([self.LABEL_DISPLAY.get(l, l) for l in labs],
                           fontsize=10)
        ax.tick_params(top=False, bottom=False, left=False, right=False)
        for s in ax.spines.values():
            s.set_visible(False)

        cbar = self.fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label(r'AUROC drop, 0% $\rightarrow$ 40% (pp)', fontsize=10)
        cbar.ax.tick_params(labelsize=9.5)

        self.add_panel_label(ax, 'e', 'Per-label degradation at 40% noise')

    def _label_mean_drop(self, lbl):
        drops = []
        for init in self.INITS:
            r0 = self.get_label(init, 0, lbl)
            r40 = self.get_label(init, 40, lbl)
            if r0 is not None and r40 is not None:
                drops.append(r0['AUROC_mean'] - r40['AUROC_mean'])
        return np.mean(drops) if drops else 0

    def _panel_f(self, ax):
        metrics = ['AUROC_mean', 'mAP_mean']
        metric_labels = ['AUROC', 'mAP']

        for m_idx, metric in enumerate(metrics):
            ax_sub = ax if m_idx == 0 else ax.twinx() if False else ax
            break

        for metric_name, linestyle, marker in [
                ('AUROC_mean', '-', 'o'),
                ('mAP_mean', '--', 's')]:
            for init in self.INITS:
                ranks = []
                for nr in self.NOISE_RATIOS:
                    vals = []
                    for other in self.INITS:
                        r = self.get_macro(other, nr)
                        vals.append((other, r[metric_name]))
                    vals.sort(key=lambda t: -t[1])  # descending
                    rank = [v[0] for v in vals].index(init) + 1
                    ranks.append(rank)
                col = self.INIT_COLOR[init]
                ax.plot(self.NOISE_RATIOS, ranks, marker=marker,
                        linestyle=linestyle, color=col, lw=2.0, ms=9,
                        mec='white', mew=1.0, zorder=3)

        ax.set_xticks(self.NOISE_RATIOS)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['1st', '2nd', '3rd'])
        ax.set_ylim(3.5, 0.5)
        ax.set_xlabel('Label noise ratio (%)')
        ax.set_ylabel('Rank among initializations')

        legend_handles = [
            Line2D([0], [0], color='#666', lw=2, marker='o',
                   mec='white', label='AUROC'),
            Line2D([0], [0], color='#666', lw=2, ls='--', marker='s',
                   mec='white', label='mAP'),
        ]
        ax.legend(handles=legend_handles, loc='lower left', fontsize=10,
                  frameon=False, handletextpad=0.5)

        self.add_panel_label(ax, 'f', 'Ranking order changes with noise')

    def save(self):
        os.makedirs(os.path.dirname(self.OUT_PNG), exist_ok=True)
        self.fig.savefig(self.OUT_PNG, dpi=300, bbox_inches='tight',
                         facecolor='white')


if __name__ == '__main__':
    Figure5().build().save()
