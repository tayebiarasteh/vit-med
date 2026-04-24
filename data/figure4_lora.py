"""
Created on April 22, 2026.
figure4_lora.py

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import FancyArrowPatch
import os


class Figure4:
    OUT_PNG  = './lora_backbone.png'

    DATASET_DISPLAY = {
        'mimic': 'MIMIC-CXR',
        'cxr14': 'ChestX-ray14',
        'chexpert': 'CheXpert',
    }
    DATASET_ORDER = ['mimic', 'cxr14', 'chexpert']

    INIT_DISPLAY = {'imgnet': 'ImageNet', 'dinov2': 'DINOv2', 'dinov3': 'DINOv3'}

    COLOR_IMGNET = '#525252'
    COLOR_DINOV2 = '#4575B4'
    COLOR_DINOV3 = '#D73027'
    INIT_COLOR   = {'imgnet': COLOR_IMGNET,
                    'dinov2': COLOR_DINOV2,
                    'dinov3': COLOR_DINOV3}

    COLOR_FULLFT = '#c7c7c7'
    COLOR_LORA   = '#525252'

    CONFIGS = [
        ('vitb',     'imgnet', 'ViT-B/16 / ImageNet'),
        ('vitb',     'dinov2', 'ViT-B/16 / DINOv2'),
        ('vitb',     'dinov3', 'ViT-B/16 / DINOv3'),
        ('convnext', 'imgnet', 'ConvNeXt-B / ImageNet'),
        ('convnext', 'dinov3', 'ConvNeXt-B / DINOv3'),
    ]

    def __init__(self):
        self.full_df = pd.read_csv(self.FULL_CSV)
        self.lora_df = pd.read_csv(self.LORA_CSV)
        self.full_macro = self.full_df[self.full_df['label'] == 'macro_avg'].copy()
        self.lora_macro = self.lora_df[self.lora_df['label'] == 'macro_avg'].copy()
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

    def get_full(self, dataset, init, backbone, size=512, label='macro_avg'):
        m = self.full_df[
            (self.full_df['dataset'] == dataset) &
            (self.full_df['initialization'] == init) &
            (self.full_df['backbone'] == backbone) &
            (self.full_df['image_size'] == size) &
            (self.full_df['label'] == label)]
        return m.iloc[0] if len(m) else None

    def get_lora(self, dataset, init, backbone, size=512, label='macro_avg'):
        m = self.lora_df[
            (self.lora_df['dataset'] == dataset) &
            (self.lora_df['initialization'] == init) &
            (self.lora_df['backbone'] == backbone) &
            (self.lora_df['image_size'] == size) &
            (self.lora_df['label'] == label)]
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
            Patch(fc=self.COLOR_FULLFT, ec='black', lw=0.6,
                  label='Full fine-tuning'),
            Patch(fc=self.COLOR_LORA, ec='black', lw=0.6, hatch='//',
                  label='LoRA'),
            Line2D([0], [0], lw=0, label='   '),
            Line2D([0], [0], marker='o', mfc='white', mec='black',
                   ms=7, lw=0, label='MIMIC-CXR'),
            Line2D([0], [0], marker='^', mfc='white', mec='black',
                   ms=7, lw=0, label='ChestX-ray14'),
            Line2D([0], [0], marker='s', mfc='white', mec='black',
                   ms=7, lw=0, label='CheXpert'),
        ]
        ax.legend(handles=items, loc='center', ncol=13, frameon=False,
                  fontsize=12, handletextpad=0.5, columnspacing=1.0)

    def _panel_a(self, ax):
        """Paired bars: per config (5), show mean AUROC across 3 datasets
        under full FT vs LoRA, with individual dataset dots overlaid."""
        x = np.arange(len(self.CONFIGS))
        width = 0.38

        ff_means, ff_cilo, ff_cihi = [], [], []
        lo_means, lo_cilo, lo_cihi = [], [], []
        ff_per_ds = {i: [] for i in range(len(self.CONFIGS))}
        lo_per_ds = {i: [] for i in range(len(self.CONFIGS))}

        for i, (bb, init, lbl) in enumerate(self.CONFIGS):
            ff_vals, ff_los, ff_his = [], [], []
            lo_vals, lo_los, lo_his = [], [], []
            for ds in self.DATASET_ORDER:
                ff = self.get_full(ds, init, bb)
                lo = self.get_lora(ds, init, bb)
                if ff is not None:
                    ff_vals.append(ff['AUROC_mean'])
                    ff_los.append(ff['AUROC_ci_low'])
                    ff_his.append(ff['AUROC_ci_high'])
                    ff_per_ds[i].append(ff['AUROC_mean'])
                if lo is not None:
                    lo_vals.append(lo['AUROC_mean'])
                    lo_los.append(lo['AUROC_ci_low'])
                    lo_his.append(lo['AUROC_ci_high'])
                    lo_per_ds[i].append(lo['AUROC_mean'])
            ff_means.append(np.mean(ff_vals))
            ff_cilo.append(np.mean(ff_los))
            ff_cihi.append(np.mean(ff_his))
            lo_means.append(np.mean(lo_vals))
            lo_cilo.append(np.mean(lo_los))
            lo_cihi.append(np.mean(lo_his))

        ff_means = np.array(ff_means)
        lo_means = np.array(lo_means)
        ff_err = np.vstack([ff_means - np.array(ff_cilo),
                            np.array(ff_cihi) - ff_means])
        lo_err = np.vstack([lo_means - np.array(lo_cilo),
                            np.array(lo_cihi) - lo_means])

        ax.bar(x - width / 2, ff_means, width, yerr=ff_err, capsize=3,
               color=self.COLOR_FULLFT, edgecolor='black', lw=0.7,
               error_kw={'elinewidth': 1.0}, label='Full FT')
        ax.bar(x + width / 2, lo_means, width, yerr=lo_err, capsize=3,
               color=self.COLOR_LORA, edgecolor='black', lw=0.7, hatch='//',
               error_kw={'elinewidth': 1.0, 'ecolor': '#333'}, label='LoRA')

        # Overlay individual dataset dots
        ds_markers = ['o', '^', 's']
        for i in range(len(self.CONFIGS)):
            for k, ds in enumerate(self.DATASET_ORDER):
                if k < len(ff_per_ds[i]):
                    ax.plot(x[i] - width / 2, ff_per_ds[i][k],
                            marker=ds_markers[k], mfc='white', mec='black',
                            ms=5, mew=0.8, zorder=5)
                if k < len(lo_per_ds[i]):
                    ax.plot(x[i] + width / 2, lo_per_ds[i][k],
                            marker=ds_markers[k], mfc='#f4f4f4', mec='black',
                            ms=5, mew=0.8, zorder=5)

        # Delta annotations
        for i in range(len(self.CONFIGS)):
            d = lo_means[i] - ff_means[i]
            y_top = max(ff_cihi[i], lo_cihi[i]) + 0.6
            ax.text(x[i], y_top, f'{d:+.1f}', ha='center', fontsize=10,
                    color='#c43030', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, _, lbl in self.CONFIGS], 
                           fontsize=10, rotation=45, ha='right')
        ax.set_ylabel('AUROC at 512 (%), pooled across 3 datasets')
        ax.set_ylim(min(lo_cilo) - 1.5, max(ff_cihi) + 2.5)

        self.add_panel_label(ax, 'a', 'LoRA reduces performance across configurations')

    def _panel_b(self, ax):
        """Heatmap of Delta AUROC (LoRA - full FT), configs x datasets."""
        data = np.zeros((len(self.CONFIGS), len(self.DATASET_ORDER)))
        sig = np.zeros_like(data, dtype=bool)
        for i, (bb, init, _) in enumerate(self.CONFIGS):
            for j, ds in enumerate(self.DATASET_ORDER):
                ff = self.get_full(ds, init, bb)
                lo = self.get_lora(ds, init, bb)
                if ff is None or lo is None:
                    continue
                data[i, j] = lo['AUROC_mean'] - ff['AUROC_mean']
                sig[i, j] = not self.ci_overlap(
                    ff['AUROC_ci_low'], ff['AUROC_ci_high'],
                    lo['AUROC_ci_low'], lo['AUROC_ci_high'])

        vmax = max(abs(data.min()), abs(data.max()))
        im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       aspect='auto')
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                star = '*' if sig[i, j] else ''
                txt_color = 'white' if abs(data[i, j]) > vmax * 0.55 else 'black'
                ax.text(j, i, f'{data[i, j]:+.1f}{star}', ha='center',
                        va='center', fontsize=10.5, color=txt_color,
                        fontweight='bold')

        ax.set_xticks(range(len(self.DATASET_ORDER)))
        ax.set_xticklabels([self.DATASET_DISPLAY[d]
                            for d in self.DATASET_ORDER], fontsize=10.5)
        ax.set_yticks(range(len(self.CONFIGS)))
        ax.set_yticklabels([lbl.replace('\n', ' ')
                            for _, _, lbl in self.CONFIGS], fontsize=10,
                           rotation=45, ha='right')
        ax.tick_params(top=False, bottom=False, left=False, right=False)
        for s in ax.spines.values():
            s.set_visible(False)

        cbar = self.fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label(r'$\Delta$AUROC: LoRA $-$ Full FT (pp)', fontsize=10.5)
        cbar.ax.tick_params(labelsize=9.5)

        self.add_panel_label(ax, 'b', 'Regime cost per configuration and dataset')

    def _panel_c(self, ax):
        """Backbone gap (CNX - ViT) at full FT vs LoRA.
        X-axis: datasets. For each dataset, group by init (ImgNet, DINOv3).
        For each init, show two points (full FT, LoRA) connected with arrow."""
        
        # Group by dataset first
        inits_to_plot = ['imgnet', 'dinov3']
        n_ds = len(self.DATASET_ORDER)
        n_init = len(inits_to_plot)
        
        # x positions: datasets with sub-positions for inits
        ds_x = np.arange(n_ds)
        init_offset = 0.18
        
        for i_ds, ds in enumerate(self.DATASET_ORDER):
            for i_init, init in enumerate(inits_to_plot):
                x_pos = ds_x[i_ds] + (i_init - 0.5) * init_offset
                
                cnx_ff = self.get_full(ds, init, 'convnext')
                vit_ff = self.get_full(ds, init, 'vitb')
                cnx_lo = self.get_lora(ds, init, 'convnext')
                vit_lo = self.get_lora(ds, init, 'vitb')
                
                if any(x is None for x in [cnx_ff, vit_ff, cnx_lo, vit_lo]):
                    continue
                    
                gap_ff = cnx_ff['AUROC_mean'] - vit_ff['AUROC_mean']
                gap_lo = cnx_lo['AUROC_mean'] - vit_lo['AUROC_mean']
                
                col = self.INIT_COLOR[init]
                
                # Full FT point (open)
                ax.plot(x_pos, gap_ff, 'o', mfc='white', mec=col, mew=1.8, 
                       ms=10, zorder=3)
                # LoRA point (filled)
                ax.plot(x_pos, gap_lo, 'o', mfc=col, mec='black', mew=0.6,
                       ms=10, zorder=3)
                # Arrow
                arr = FancyArrowPatch((x_pos, gap_ff), (x_pos, gap_lo),
                                     arrowstyle='->,head_length=6,head_width=4',
                                     color=col, lw=1.7, zorder=2, mutation_scale=1)
                ax.add_patch(arr)
        
        ax.axhline(0, color='black', lw=1.0, ls='--', zorder=1)
        
        ax.set_xticks(ds_x)
        ax.set_xticklabels([self.DATASET_DISPLAY[d] 
                           for d in self.DATASET_ORDER], fontsize=10.5)
        ax.set_ylabel(r'Backbone gap: ConvNeXt-B $-$ ViT-B/16 (pp)')
        ax.set_ylim(0, None)  # Start from 0
        
        leg_handles = [
            Line2D([0], [0], marker='o', mfc='white', mec='#333',
                   mew=1.8, ms=9, lw=0, label='Full FT'),
            Line2D([0], [0], marker='o', mfc='#333', mec='black',
                   mew=0.6, ms=9, lw=0, label='LoRA'),
        ]
        ax.legend(handles=leg_handles, loc='upper left', fontsize=10,
                 frameon=False, handletextpad=0.5)
        
        self.add_panel_label(ax, 'c', 'Backbone gap does not narrow under LoRA')

    def _panel_d(self, ax):

        backbones = ['convnext', 'vitb']
        n_ds = len(self.DATASET_ORDER)
        n_bb = len(backbones)
        
        ds_x = np.arange(n_ds)
        bb_offset = 0.18
        
        for i_ds, ds in enumerate(self.DATASET_ORDER):
            for i_bb, bb in enumerate(backbones):
                x_pos = ds_x[i_ds] + (i_bb - 0.5) * bb_offset
                
                d3_ff = self.get_full(ds, 'dinov3', bb)
                ig_ff = self.get_full(ds, 'imgnet', bb)
                d3_lo = self.get_lora(ds, 'dinov3', bb)
                ig_lo = self.get_lora(ds, 'imgnet', bb)
                
                if any(x is None for x in [d3_ff, ig_ff, d3_lo, ig_lo]):
                    continue
                
                gap_ff = d3_ff['AUROC_mean'] - ig_ff['AUROC_mean']
                gap_lo = d3_lo['AUROC_mean'] - ig_lo['AUROC_mean']
                
                col = self.COLOR_DINOV3
                marker = 's' if bb == 'convnext' else 'D'
                
                ax.plot(x_pos, gap_ff, marker, mfc='white', mec=col, mew=1.8,
                       ms=10, zorder=3)
                ax.plot(x_pos, gap_lo, marker, mfc=col, mec='black', mew=0.6,
                       ms=10, zorder=3)
                arr = FancyArrowPatch((x_pos, gap_ff), (x_pos, gap_lo),
                                     arrowstyle='->,head_length=6,head_width=4',
                                     color=col, lw=1.7, zorder=2, mutation_scale=1)
                ax.add_patch(arr)
        
        ax.axhline(0, color='black', lw=1.0, ls='--', zorder=1)
        
        ax.set_xticks(ds_x)
        ax.set_xticklabels([self.DATASET_DISPLAY[d]
                           for d in self.DATASET_ORDER], fontsize=10.5)
        ax.set_ylabel(r'DINOv3 advantage: DINOv3 $-$ ImageNet (pp)')
        ax.set_ylim(-0.5, None)  # Start from -0.5 to show negative values
        
        # Legend
        leg_handles = [
            Line2D([0], [0], marker='o', mfc='white', mec=self.COLOR_DINOV3,
                   mew=1.8, ms=9, lw=0, label='Full FT'),
            Line2D([0], [0], marker='o', mfc=self.COLOR_DINOV3, mec='black',
                   mew=0.6, ms=9, lw=0, label='LoRA'),
        ]
        ax.legend(handles=leg_handles, loc='upper left', fontsize=10,
                 frameon=False, handletextpad=0.5)
        
        self.add_panel_label(ax, 'd', 'Advantage persists but varies by configuration')

    def _panel_e(self, ax):
        """For each dataset, show distribution of per-label
        Delta AUROC (LoRA - full FT) across all configurations."""
        rng = np.random.default_rng(42)
        x_positions = np.arange(len(self.DATASET_ORDER))

        for i, ds in enumerate(self.DATASET_ORDER):
            labs = self.full_df[(self.full_df['dataset'] == ds) &
                                (self.full_df['label'] != 'macro_avg')
                                ]['label'].unique()
            all_deltas = []
            all_colors = []
            all_markers = []
            for bb in ['convnext', 'vitb']:
                mk = 's' if bb == 'convnext' else 'D'
                for init in ['imgnet', 'dinov2', 'dinov3']:
                    if init == 'dinov2' and bb == 'convnext':
                        continue
                    for lbl in labs:
                        ff = self.get_full(ds, init, bb, label=lbl)
                        lo = self.get_lora(ds, init, bb, label=lbl)
                        if ff is None or lo is None:
                            continue
                        all_deltas.append(lo['AUROC_mean'] - ff['AUROC_mean'])
                        all_colors.append(self.INIT_COLOR[init])
                        all_markers.append(mk)

            jitter = rng.uniform(-0.28, 0.28, size=len(all_deltas))
            for d, j, c, mk in zip(all_deltas, jitter, all_colors, all_markers):
                ax.scatter(x_positions[i] + j, d, s=38, marker=mk,
                           facecolors=c, edgecolors='white', lw=0.6,
                           alpha=0.75, zorder=3)

            # Median line
            med = np.median(all_deltas)
            ax.plot([x_positions[i] - 0.33, x_positions[i] + 0.33],
                    [med, med], color='black', lw=2.2, zorder=4)
            ax.text(x_positions[i] + 0.36, med, f'{med:+.1f}',
                    va='center', ha='left', fontsize=10.5, fontweight='bold')

        ax.axhline(0, color='black', lw=0.8, ls='--', zorder=1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([self.DATASET_DISPLAY[d]
                            for d in self.DATASET_ORDER], fontsize=11)
        ax.set_ylabel(r'$\Delta$AUROC: LoRA $-$ Full FT (pp)' + '\n'
                      + 'per label')
        self.add_panel_label(ax, 'e', 'Regime cost is heterogeneous across labels')

    def _panel_f(self, ax):
        ds_markers = {'mimic': 'o', 'cxr14': '^', 'chexpert': 's'}

        for i, (bb, init, _) in enumerate(self.CONFIGS):
            col = self.INIT_COLOR[init]
            edge = 'black' if bb == 'convnext' else '#333'
            edge_w = 1.6 if bb == 'convnext' else 0.8
            for ds in self.DATASET_ORDER:
                r = self.get_lora(ds, init, bb)
                if r is None:
                    continue
                mk = ds_markers[ds]
                ax.scatter(r['AUROC_mean'], r['mAP_mean'], s=140, marker=mk,
                           facecolors=col, edgecolors=edge, linewidths=edge_w,
                           alpha=0.85, zorder=3)

        for ds in self.DATASET_ORDER:
            xs, ys = [], []
            for bb, init, _ in self.CONFIGS:
                r = self.get_lora(ds, init, bb)
                if r is not None:
                    xs.append(r['AUROC_mean'])
                    ys.append(r['mAP_mean'])
            if len(xs) >= 2:
                xs_a = np.array(xs)
                ys_a = np.array(ys)
                coef = np.polyfit(xs_a, ys_a, 1)
                order = np.argsort(xs_a)
                ax.plot(xs_a[order], np.polyval(coef, xs_a[order]),
                        color='#999', lw=0.8, ls=':', zorder=1)

        try:
            from scipy.stats import spearmanr
            pass
        except ImportError:
            pass

        ax.set_xlabel('AUROC under LoRA (%)')
        ax.set_ylabel('mAP under LoRA (%)')

        ds_handles = [
            Line2D([0], [0], marker=ds_markers[d], mfc='#bbb', mec='black',
                   ms=9, lw=0, label=self.DATASET_DISPLAY[d])
            for d in self.DATASET_ORDER]
        ax.legend(handles=ds_handles, loc='lower right', fontsize=9.5,
                  frameon=False, handletextpad=0.4, labelspacing=0.25)

        self.add_panel_label(ax, 'f', 'Ranking is metric-robust under LoRA')

    def save(self):
        os.makedirs(os.path.dirname(self.OUT_PNG), exist_ok=True)
        self.fig.savefig(self.OUT_PNG, dpi=300, bbox_inches='tight',
                         facecolor='white')


if __name__ == '__main__':
    Figure4().build().save()
