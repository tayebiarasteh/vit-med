"""
Created on April 22, 2026.
figure6_external.py

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch


class Figure6:
    OUT_PNG = './external_results.png'

    COLOR_IMGNET = '#525252'
    COLOR_DINOV2 = '#4575B4'
    COLOR_DINOV3 = '#D73027'
    COLOR_FROZEN = '#8B0000'
    COLOR_FT = '#D73027'

    INIT_COLOR = {'imgnet': COLOR_IMGNET,
                  'dinov2': COLOR_DINOV2,
                  'dinov3': COLOR_DINOV3}
    INIT_DISPLAY = {'imgnet': 'ImageNet', 'dinov2': 'DINOv2', 'dinov3': 'DINOv3'}

    CONFIGS = [
        ('vitb',     'imgnet', 'ViT-B/16 / ImageNet'),
        ('vitb',     'dinov2', 'ViT-B/16 / DINOv2'),
        ('vitb',     'dinov3', 'ViT-B/16 / DINOv3'),
        ('convnext', 'imgnet', 'ConvNeXt-B / ImageNet'),
        ('convnext', 'dinov3', 'ConvNeXt-B / DINOv3'),
    ]
    DATASET_ORDER = ['cxr14', 'chexpert']
    DATASET_DISPLAY = {
        'cxr14': 'ChestX-ray14', 'chexpert': 'CheXpert',
        'mimic': 'MIMIC-CXR', 'vindr': 'VinDr-CXR',
        'padchest': 'PadChest', 'UKA': 'UKA-CXR', 'pedi': 'Pedi-CXR',
    }

    FROZEN_DS_ORDER = ['pedi', 'vindr', 'cxr14', 'padchest', 'chexpert',
                       'mimic', 'UKA']

    def __init__(self):
        self.ext = pd.read_csv(self.EXT_CSV)
        self.frozen = pd.read_csv(self.FROZEN_CSV)
        self.full = pd.read_csv(self.FULL_CSV)
        self.em = self.ext[self.ext['label'] == 'macro_avg'].copy()
        self.fm = self.full[self.full['label'] == 'macro_avg'].copy()
        self.frm = self.frozen[self.frozen['label'] == 'macro_avg'].copy()
        self._set_rcparams()
        self.fig = None

    @staticmethod
    def _set_rcparams():
        plt.rcParams.update({
            'font.family': 'DejaVu Sans', 'font.size': 12,
            'axes.labelsize': 13, 'axes.titlesize': 13,
            'xtick.labelsize': 11, 'ytick.labelsize': 11,
            'legend.fontsize': 10.5,
            'axes.spines.top': False, 'axes.spines.right': False,
            'axes.linewidth': 1.1,
            'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
            'axes.grid': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        })

    def get_ext(self, ds, eval_type, init, bb, size):
        m = self.em[(self.em['dataset'] == ds) &
                    (self.em['evaluation_type'] == eval_type) &
                    (self.em['initialization'] == init) &
                    (self.em['backbone'] == bb) &
                    (self.em['image_size'] == size)]
        return m.iloc[0] if len(m) else None

    def get_frozen(self, ds, size):
        m = self.frm[(self.frm['dataset'] == ds) &
                     (self.frm['image_size'] == size)]
        return m.iloc[0] if len(m) else None

    def get_best_ft(self, ds, size):
        """Return (best_auroc, bb, init) for best full-FT config at given
        dataset/size."""
        m = self.fm[(self.fm['dataset'] == ds) &
                    (self.fm['image_size'] == size)]
        if not len(m):
            return None, None, None
        best = m.loc[m['AUROC_mean'].idxmax()]
        return best['AUROC_mean'], best['backbone'], best['initialization']

    @staticmethod
    def add_panel_label(ax, letter, title='', x=-0.14, y=1.05):
        ax.text(x, y, letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='bottom', ha='left')
        if title:
            ax.text(x + 0.08, y, title, transform=ax.transAxes,
                    fontsize=13, fontweight='normal', va='bottom', ha='left')

    def build(self):
        self.fig = plt.figure(figsize=(17, 12.5))

        gs_leg = self.fig.add_gridspec(1, 1, left=0.04, right=0.99,
                                       top=0.997, bottom=0.955)
        gs = self.fig.add_gridspec(2, 3, hspace=0.48, wspace=0.36,
                                   left=0.06, right=0.98,
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
            # Init colors (same as previous figures)
            Line2D([0], [0], marker='o', color=self.COLOR_IMGNET,
                   mfc=self.COLOR_IMGNET, ms=9, lw=2.2, label='ImageNet'),
            Line2D([0], [0], marker='o', color=self.COLOR_DINOV2,
                   mfc=self.COLOR_DINOV2, ms=9, lw=2.2, label='DINOv2'),
            Line2D([0], [0], marker='o', color=self.COLOR_DINOV3,
                   mfc=self.COLOR_DINOV3, ms=9, lw=2.2, label='DINOv3'),
            Line2D([0], [0], lw=0, label='    '),
            # Backbone shapes
            Line2D([0], [0], marker='s', color='#808080',
                   mec='#808080', ms=9, lw=0, label='ConvNeXt-B'),
            Line2D([0], [0], marker='D', color='#808080',
                   mec='#808080', ms=8, lw=0, label='ViT-B/16'),
            Line2D([0], [0], lw=0, label='    '),
            Patch(fc='#cccccc', ec='black', lw=0.6, label='Internal'),
            Patch(fc='#666666', ec='black', lw=0.6, label='External (MIMIC-trained)'),
            Line2D([0], [0], lw=0, label='    '),
            Patch(fc=self.COLOR_FROZEN, ec='black', lw=0.6, hatch='//',
                  label='Frozen DINOv3-7B'),
            Patch(fc=self.COLOR_FT, ec='black', lw=0.6,
                  label='Best fine-tuned'),
        ]
        ax.legend(handles=items, loc='center', ncol=13, frameon=False,
                  fontsize=12, handletextpad=0.5, columnspacing=1.0)

    def _panel_a(self, ax):
        """Dumbbell showing internal -> external AUROC drop at 512.
        One row per (dataset, config). Clear arrows show drop direction."""
        items = []
        for ds in self.DATASET_ORDER:
            for bb, init, lbl in self.CONFIGS:
                rint = self.get_ext(ds, 'internal', init, bb, 512)
                rext = self.get_ext(ds, 'trained on mimic', init, bb, 512)
                if rint is not None and rext is not None:
                    items.append((ds, bb, init, lbl,
                                  rint['AUROC_mean'], rext['AUROC_mean']))

        y_positions = np.arange(len(items))[::-1].astype(float)

        ds_change_idx = None
        for i, (ds, *_) in enumerate(items):
            if i > 0 and items[i][0] != items[i-1][0]:
                ds_change_idx = i
                break

        for yp, (ds, bb, init, lbl, a_int, a_ext) in zip(y_positions, items):
            col = self.INIT_COLOR[init]
            marker = 's' if bb == 'convnext' else 'D'
            ax.plot(a_int, yp, marker, mfc='white', mec=col, mew=1.8, ms=10,
                    zorder=3)
            ax.plot(a_ext, yp, marker, mfc=col, mec='black', mew=0.6, ms=10,
                    zorder=3)
            arr = FancyArrowPatch((a_int, yp), (a_ext, yp),
                                  arrowstyle='->,head_length=6,head_width=4',
                                  color=col, lw=1.6, zorder=2,
                                  mutation_scale=1)
            ax.add_patch(arr)

        if ds_change_idx is not None:
            sep_y = y_positions[ds_change_idx] + 0.5
            ax.axhline(sep_y, color='#ccc', lw=0.8, ls='-', zorder=1)

        ylabs = [f'{lbl}' for ds, bb, init, lbl, _, _ in items]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(ylabs, fontsize=9)

        ax.set_xlabel('AUROC at 512 (%)')

        cxr_items = [(yp, it) for yp, it in zip(y_positions, items)
                     if it[0] == 'cxr14']
        chx_items = [(yp, it) for yp, it in zip(y_positions, items)
                     if it[0] == 'chexpert']
        cxr_top = max(yp for yp, _ in cxr_items) + 0.3
        chx_mid = np.mean([yp for yp, _ in chx_items])

        xlim = ax.get_xlim()
        x_cxr = xlim[1] - 0.15
        ax.text(x_cxr, cxr_top, 'ChestX-ray14',
                ha='right', va='bottom', fontsize=9.5, style='italic',
                color='#555', fontweight='bold')
        x_chx = xlim[0] + 0.15
        ax.text(x_chx, chx_mid, 'CheXpert',
                ha='left', va='center', fontsize=9.5, style='italic',
                color='#555', fontweight='bold')

        leg_handles = [
            Line2D([0], [0], marker='o', mfc='white', mec='#333', mew=1.8,
                   ms=9, lw=0, label='Internal'),
            Line2D([0], [0], marker='o', mfc='#333', mec='black', mew=0.6,
                   ms=9, lw=0, label='External'),
        ]
        ax.legend(handles=leg_handles, loc='lower left', fontsize=9.5,
                  frameon=False, handletextpad=0.4)

        self.add_panel_label(ax, 'a',
                             'External drop is modest across configurations')

    def _panel_b(self, ax):
        """For each (dataset, config), show dumbbell: 224 AUROC -> 512 AUROC.
        Grouped by dataset."""
        items = []
        for ds in self.DATASET_ORDER:
            for bb, init, lbl in self.CONFIGS:
                r224 = self.get_ext(ds, 'trained on mimic', init, bb, 224)
                r512 = self.get_ext(ds, 'trained on mimic', init, bb, 512)
                if r224 is not None and r512 is not None:
                    items.append((bb, init, ds, lbl,
                                  r224['AUROC_mean'], r512['AUROC_mean']))

        y_positions = np.arange(len(items))[::-1].astype(float)

        for yp, (bb, init, ds, lbl, a224, a512) in zip(y_positions, items):
            col = self.INIT_COLOR[init]
            marker = 's' if bb == 'convnext' else 'D'
            ax.plot(a224, yp, marker, mfc='white', mec=col, mew=1.8, ms=10,
                    zorder=3)
            ax.plot(a512, yp, marker, mfc=col, mec='black', mew=0.6, ms=10,
                    zorder=3)
            arr = FancyArrowPatch((a224, yp), (a512, yp),
                                  arrowstyle='->,head_length=6,head_width=4',
                                  color=col, lw=1.6, zorder=2,
                                  mutation_scale=1)
            ax.add_patch(arr)

        ds_change_idx = None
        for i in range(1, len(items)):
            if items[i][2] != items[i-1][2]:
                ds_change_idx = i
                break
        if ds_change_idx is not None:
            sep_y = y_positions[ds_change_idx] + 0.5
            ax.axhline(sep_y, color='#ccc', lw=0.8, ls='-', zorder=1)

        ylabs = [f'{lbl}' for bb, init, ds, lbl, _, _ in items]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(ylabs, fontsize=9)

        ax.set_xlabel('External AUROC (%)')

        cxr_top = max(yp for yp, it in zip(y_positions, items)
                      if it[2] == 'cxr14') + 0.3
        chx_mid = np.mean([yp for yp, it in zip(y_positions, items)
                           if it[2] == 'chexpert'])

        xlim = ax.get_xlim()
        x_cxr = xlim[1] - 0.1
        ax.text(x_cxr, cxr_top, 'ChestX-ray14',
                ha='right', va='bottom', fontsize=9.5, style='italic',
                color='#555', fontweight='bold')
        x_chx = xlim[0] + 0.1
        ax.text(x_chx, chx_mid, 'CheXpert',
                ha='left', va='center', fontsize=9.5, style='italic',
                color='#555', fontweight='bold')

        leg_handles = [
            Line2D([0], [0], marker='o', mfc='white', mec='#333', mew=1.8,
                   ms=9, lw=0, label='224 x 224'),
            Line2D([0], [0], marker='o', mfc='#333', mec='black', mew=0.6,
                   ms=9, lw=0, label='512 x 512'),
        ]
        ax.legend(handles=leg_handles, loc='lower left', fontsize=9.5,
                  frameon=False, handletextpad=0.4)

        self.add_panel_label(ax, 'b',
                             'Resolution advantage persists externally')

    def _panel_c(self, ax):
        """For each config, show internal-external AUROC gap for each
        target dataset. Grouped bars."""
        n_cfg = len(self.CONFIGS)
        n_ds = len(self.DATASET_ORDER)
        bar_width = 0.35

        x = np.arange(n_cfg)

        ds_hatches = {'cxr14': '', 'chexpert': '///'}
        ds_labels = [self.DATASET_DISPLAY[d] for d in self.DATASET_ORDER]

        for j, ds in enumerate(self.DATASET_ORDER):
            offsets = (j - 0.5) * bar_width
            gaps = []
            for bb, init, lbl in self.CONFIGS:
                rint = self.get_ext(ds, 'internal', init, bb, 512)
                rext = self.get_ext(ds, 'trained on mimic', init, bb, 512)
                if rint is not None and rext is not None:
                    gaps.append(rint['AUROC_mean'] - rext['AUROC_mean'])
                else:
                    gaps.append(0)
            cols = [self.INIT_COLOR[init] for _, init, _ in self.CONFIGS]
            bars = ax.bar(x + offsets, gaps, bar_width,
                          color=cols, edgecolor='black', lw=0.6,
                          hatch=ds_hatches[ds], label=ds_labels[j])
            for xi, g in zip(x + offsets, gaps):
                ax.text(xi, g + 0.05, f'{g:.2f}', ha='center', va='bottom',
                        fontsize=8.5, color='black')

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, _, lbl in self.CONFIGS],
                           fontsize=9.5, rotation=45, ha='right')
        ax.set_ylabel('Internal - External AUROC (pp)')
        ax.set_ylim(0, 3.5)

        hatch_legend = [
            Patch(fc='white', ec='black', lw=0.6, hatch='',
                  label='ChestX-ray14'),
            Patch(fc='white', ec='black', lw=0.6, hatch='///',
                  label='CheXpert'),
        ]
        ax.legend(handles=hatch_legend, loc='upper left', fontsize=9.5,
                  frameon=False, handletextpad=0.5)

        self.add_panel_label(ax, 'c', 'DINOv3 generalizes comparably')

    def _panel_d(self, ax):
        for ds in self.DATASET_ORDER:
            marker = 'o' if ds == 'cxr14' else 's'
            xs, ys = [], []
            for bb, init, lbl in self.CONFIGS:
                r = self.get_ext(ds, 'trained on mimic', init, bb, 512)
                if r is not None:
                    col = self.INIT_COLOR[init]
                    ec = 'black' if bb == 'convnext' else col
                    lw = 1.4 if bb == 'convnext' else 0.8
                    ax.plot(r['AUROC_mean'], r['mAP_mean'], marker,
                            mfc=col, mec=ec, mew=lw, ms=12, zorder=3)
                    xs.append(r['AUROC_mean'])
                    ys.append(r['mAP_mean'])

            if len(xs) >= 2:
                xs_arr = np.array(xs)
                ys_arr = np.array(ys)
                coef = np.polyfit(xs_arr, ys_arr, 1)
                xr = np.linspace(xs_arr.min() - 0.2, xs_arr.max() + 0.2, 50)
                ax.plot(xr, np.polyval(coef, xr), ls=':', color='#999',
                        lw=1.2, zorder=1)

        ax.set_xlabel('External AUROC (%)')
        ax.set_ylabel('External mAP (%)')

        shape_leg = [
            Line2D([0], [0], marker='o', mfc='#808080', mec='black', mew=0.6,
                   ms=10, lw=0, label='ChestX-ray14'),
            Line2D([0], [0], marker='s', mfc='#808080', mec='black', mew=0.6,
                   ms=10, lw=0, label='CheXpert'),
        ]
        ax.legend(handles=shape_leg, loc='lower right', fontsize=9.5,
                  frameon=False, handletextpad=0.4)

        self.add_panel_label(ax, 'd', 'Ranking is metric-robust externally')

    def _panel_e(self, ax):
        """For each of 7 datasets, show frozen vs best-FT AUROC at 224 and 512.
        Layout: 7 dataset groups, each with 4 bars (fr224, ft224, fr512, ft512)."""
        n_ds = len(self.FROZEN_DS_ORDER)
        group_width = 0.75
        bar_width = group_width / 4
        gap_between_sizes = bar_width * 0.3

        x = np.arange(n_ds)

        for i, ds in enumerate(self.FROZEN_DS_ORDER):
            fr224 = self.get_frozen(ds, 224)
            fr512 = self.get_frozen(ds, 512)
            ft224, _, _ = self.get_best_ft(ds, 224)
            ft512, _, _ = self.get_best_ft(ds, 512)
            if any(v is None for v in [fr224, fr512, ft224, ft512]):
                continue

            left_center = i - bar_width - gap_between_sizes / 2
            right_center = i + bar_width + gap_between_sizes / 2
            # 224 cluster
            ax.bar(left_center - bar_width / 2, fr224['AUROC_mean'], bar_width,
                   color=self.COLOR_FROZEN, edgecolor='black', lw=0.5,
                   hatch='//')
            ax.bar(left_center + bar_width / 2, ft224, bar_width,
                   color=self.COLOR_FT, edgecolor='black', lw=0.5)
            # 512 cluster
            ax.bar(right_center - bar_width / 2, fr512['AUROC_mean'],
                   bar_width, color=self.COLOR_FROZEN, edgecolor='black',
                   lw=0.5, hatch='//')
            ax.bar(right_center + bar_width / 2, ft512, bar_width,
                   color=self.COLOR_FT, edgecolor='black', lw=0.5)

            ax.text(left_center, 67.5, '224', ha='center', va='top',
                    fontsize=7.5, color='#444', fontweight='bold')
            ax.text(right_center, 67.5, '512', ha='center', va='top',
                    fontsize=7.5, color='#444', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([self.DATASET_DISPLAY[d]
                            for d in self.FROZEN_DS_ORDER],
                           fontsize=9.5, rotation=30, ha='right')
        ax.tick_params(axis='x', pad=12)
        ax.set_ylabel('AUROC (%)')
        ax.set_ylim(67, 93)
        self.add_panel_label(ax, 'e',
                             'Frozen DINOv3-7B trails best fine-tuned')

    def _panel_f(self, ax):
        y_positions = np.arange(len(self.FROZEN_DS_ORDER))[::-1].astype(float)

        gaps_224, gaps_512 = [], []
        for ds in self.FROZEN_DS_ORDER:
            fr224 = self.get_frozen(ds, 224)
            fr512 = self.get_frozen(ds, 512)
            ft224, _, _ = self.get_best_ft(ds, 224)
            ft512, _, _ = self.get_best_ft(ds, 512)
            if any(x is None for x in [fr224, fr512, ft224, ft512]):
                gaps_224.append(None)
                gaps_512.append(None)
                continue
            gaps_224.append(ft224 - fr224['AUROC_mean'])
            gaps_512.append(ft512 - fr512['AUROC_mean'])

        for yp, g224, g512, ds in zip(y_positions, gaps_224, gaps_512,
                                      self.FROZEN_DS_ORDER):
            if g224 is None or g512 is None:
                continue
            ax.plot(g224, yp, 'o', mfc='white', mec='#333', mew=1.8, ms=10,
                    zorder=3)
            ax.plot(g512, yp, 'o', mfc='#333', mec='black', mew=0.6, ms=10,
                    zorder=3)
            # Arrow
            direction = '->' if g512 < g224 else '->'
            col = '#228B22' if g512 < g224 else '#D73027'  # green=narrowed
            arr = FancyArrowPatch((g224, yp), (g512, yp),
                                  arrowstyle='->,head_length=6,head_width=4',
                                  color=col, lw=1.6, zorder=2,
                                  mutation_scale=1)
            ax.add_patch(arr)
            # Annotate change
            change = g512 - g224
            ax.text(max(g224, g512) + 0.15, yp, f'{change:+.1f}',
                    ha='left', va='center', fontsize=9, color=col,
                    fontweight='bold')

        ax.set_yticks(y_positions)
        ax.set_yticklabels([self.DATASET_DISPLAY[d]
                            for d in self.FROZEN_DS_ORDER], fontsize=10)
        ax.set_xlabel('Best fine-tuned AUROC - Frozen AUROC (pp)')
        ax.set_xlim(2, 9)

        leg_handles = [
            Line2D([0], [0], marker='o', mfc='white', mec='#333', mew=1.8,
                   ms=9, lw=0, label='224 x 224'),
            Line2D([0], [0], marker='o', mfc='#333', mec='black', mew=0.6,
                   ms=9, lw=0, label='512 x 512'),
        ]
        ax.legend(handles=leg_handles, loc='lower right', fontsize=9.5,
                  frameon=False, handletextpad=0.4)

        self.add_panel_label(ax, 'f',
                             'Resolution narrows the frozen-FT gap')

    def save(self):
        os.makedirs(os.path.dirname(self.OUT_PNG), exist_ok=True)
        self.fig.savefig(self.OUT_PNG, dpi=300, bbox_inches='tight',
                         facecolor='white')


if __name__ == '__main__':
    Figure6().build().save()
