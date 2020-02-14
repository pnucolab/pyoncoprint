import numpy as np
import pandas as pd

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

class OncoPrint:
    def __init__(self, recurrence_matrix, zorders=None, genes=None, samples=None, seperator=","):
        if isinstance(recurrence_matrix, pd.DataFrame):
            if samples is None:
                samples = recurrence_matrix.columns
            if genes is None:
                genes = recurrence_matrix.index
            mat = recurrence_matrix.to_numpy()
        else:
            mat = recurrence_matrix

        if genes is None:
            genes = np.array(["Gene %d"%i for i in range(1, arr.shape[0] + 1)])
        if samples is None:
            samples = np.array(["Sample %d"%i for i in range(1, arr.shape[1] + 1)])
        
        dedup_mat = []
        for g in list(np.unique(genes)):
            rows = mat[genes == g, :]
            joined_row = rows[0]
            for ridx in range(1, len(rows)):
                for cidx in range(len(samples)):
                    if isinstance(joined_row[cidx], str) and isinstance(rows[ridx][cidx], str):
                        joined_row[cidx] += seperator + rows[ridx][cidx]
                    elif isinstance(rows[ridx][cidx], str):
                        joined_row[cidx] = rows[ridx][cidx]
            dedup_mat.append(joined_row)
        mat = np.array(dedup_mat)
        genes = np.unique(genes)
        
        mutation_types = set()
        cntmat = np.zeros_like(mat, dtype=int)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if isinstance(mat[i,j], str):
                    mutations = np.unique(mat[i,j].split(seperator))
                    for mut in mutations:
                        mutation_types.add(mut)
                    cntmat[i,j] = len(mutations)

        if zorders:
            self.mutation_types = zorders
        else:
            self.mutation_types = sorted(mutation_types)
        
        self.seperator = seperator
        self.mat = mat
        self.genes = genes
        self.samples = samples
        self.count_mat = cntmat
        
        
    def _sort_default(self):
        sorted_gene_indices = np.argsort(np.sum(self.count_mat.astype(bool), axis=1))[::-1] # gene order
        sorted_sample_indices = []
        for gidx in range(len(sorted_gene_indices)):
            r = sorted_gene_indices[gidx]
            sidx_hasmuts = np.where(self.count_mat[r, :] > 0)[0]
            cnts = self.count_mat[:, sidx_hasmuts][r, :]
            for cnt in np.unique(cnts)[::-1]: # np.unique returns sorted array
                sidx_cntsorted = sidx_hasmuts[cnts == cnt]
                muts = self.mat[:, sidx_cntsorted][r, :]
                mut_types, mut_freqs = np.unique(muts, return_counts=True)
                for mut in mut_types[np.argsort(mut_freqs)[::-1]]:
                    sidx_sorted = sidx_cntsorted[muts == mut]
                    for sidx in sidx_sorted:
                        if not sidx in sorted_sample_indices:
                            sorted_sample_indices.append(sidx)
        sorted_sample_indices = np.array(sorted_sample_indices)
        return sorted_gene_indices, sorted_sample_indices
        
    def oncoprint(self, markers, title="", title_fontsize=30, sort_method='default', figsize=[50, 20], width_ratios=[8, 1, 1], height_ratios=[1, 5], background_color="#dddddd", gap=0.2, leftaxis_template="{0:.0%}"):
        
        if sort_method == 'unsorted':
            sorted_gene_indices, sorted_sample_indices = np.arange(self.mat.shape[0]), np.arange(self.mat.shape[1])
        elif sort_method != 'default':
            print("Warning: sort method other than default is not supported yet.")
            sorted_gene_indices, sorted_sample_indices = self._sort_default()
        else:
            sorted_gene_indices, sorted_sample_indices = self._sort_default()
        
        sorted_genes = self.genes[sorted_gene_indices]
        sorted_samples = self.samples[sorted_sample_indices]
        sorted_mat = self.mat[sorted_gene_indices, :][:, sorted_sample_indices]
        
        background_length = 1 - gap
        backgrounds = []
        fill_mutations = defaultdict(lambda: [])
        scatter_mutations = defaultdict(lambda: [[], []])
        
        stacked_counts_top = np.zeros([len(self.mutation_types), sorted_mat.shape[1]])
        stacked_counts_right = np.zeros([len(self.mutation_types), sorted_mat.shape[0]])
        counts_left = np.zeros(sorted_mat.shape[0])
        for i in range(sorted_mat.shape[0]):
            for j in range(sorted_mat.shape[1]):
                rect = Rectangle((j - background_length/2.0, i - background_length/2.0), background_length, background_length)
                backgrounds.append(rect)
                if isinstance(sorted_mat[i,j], str):
                    counts_left[i] += 1
                    for mut in sorted_mat[i,j].split(self.seperator):
                        stacked_counts_top[self.mutation_types.index(mut), j] += 1
                        stacked_counts_right[self.mutation_types.index(mut), i] += 1
                        ms = markers[mut]
                        if ms['marker'] == 'fill' or ms['marker'] == 'rect':
                            full_length = 1 - gap
                            w = full_length * ms.get('width', 1)
                            h = full_length * ms.get('height', 1)
                            mut_marker = Rectangle((j - w * 0.5, i - h * 0.5), w, h)
                            fill_mutations[mut].append(mut_marker)
                        else:
                            scatter_mutations[mut][0].append(j)
                            scatter_mutations[mut][1].append(i)

        f = plt.figure(figsize=figsize)

        gs = gridspec.GridSpec(2, 3, width_ratios=width_ratios, height_ratios=height_ratios)
        gs.update(wspace=0.15, hspace=0.05)
        
        #f, ((ax_top, ax_empty1, ax_empty2), (ax, ax_right, ax_legend)) = plt.subplots(2, 3, figsize=figsize, gridspec_kw={
        #    'width_ratios': width_ratios,
        #    'height_ratios': height_ratios,
        #})
        
        ax_top = plt.subplot(gs[0])
        ax_empty1 = plt.subplot(gs[1])
        ax_empty2 = plt.subplot(gs[2])
        ax = plt.subplot(gs[3])
        ax_right = plt.subplot(gs[4])
        ax_legend = plt.subplot(gs[5])
                
        if title != "":
            ttl = f.suptitle(title, fontsize=title_fontsize)
        
        pc = PatchCollection(backgrounds, facecolor=background_color)
        ax.add_collection(pc)

        for mut, patches in fill_mutations.items():
            col = markers[mut].get('color', "red")
            pc = PatchCollection(patches, facecolor=col)
            ax.add_collection(pc)

        for mut, (x, y) in scatter_mutations.items():
            ms = markers[mut]
            m = ms.get('marker', '.')
            c = ms.get('color', "red")
            s = ms.get('size', 50)
            ax.scatter(x, y, marker=m, c=c, s=s)

        ax_xticks = range(len(sorted_samples))
        ax_yticks = range(len(sorted_genes))

        ax_top.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_right.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        bottom = np.zeros(sorted_mat.shape[1])
        for idx, cnts in enumerate(stacked_counts_top):
            col = markers[self.mutation_types[idx]]['color']
            ax_top.bar(ax_xticks, cnts, color=col, bottom=bottom)
            bottom += cnts
            
        left = np.zeros(sorted_mat.shape[0])
        for idx, cnts in enumerate(stacked_counts_right):
            col = markers[self.mutation_types[idx]]['color']
            ax_right.barh(ax_yticks, cnts, color=col, left=left)
            left += cnts
            
        legend_elements = []
        for mut in self.mutation_types:
            ms = markers[mut]
            col = ms['color']
            mk = ms['marker']
            if mk == 'fill' or mk == 'rect':
                el = Patch(facecolor=col, edgecolor=col, label=mut)
            else:
                el = Line2D([0], [0], color='#ffffff00', markerfacecolor=col, markeredgecolor=col, markeredgewidth=2, marker=mk, markersize=ms.get('size', 20), label=mut)
            legend_elements.append(el)
        leg = ax_legend.legend(handles=legend_elements, loc='center', prop={'size': 20})
        leg_fr = leg.get_frame()
        
        leg_fr.set_edgecolor('none')
        leg_fr.set_facecolor('none')
        leg_fr.set_linewidth(0.0)

        ax.set_xticks(ax_xticks)
        ax.set_xticklabels(sorted_samples)
        ax.tick_params(axis='x', rotation=90)

        ax.set_yticks(ax_yticks)
        ax.set_yticklabels([leftaxis_template.format(e/float(sorted_mat.shape[1])) for e in counts_left])

        ax2 = ax.twinx()
        ax2.set_yticks(ax_yticks)
        ax2.set_yticklabels(sorted_genes)

        ax_xlim = [-background_length/2.0, sorted_mat.shape[1] - 1 + background_length/2.0]
        ax_ylim = [sorted_mat.shape[0] - 1 + background_length/2.0, -background_length/2.0]

        ax.set_xlim(ax_xlim)
        ax.set_ylim(ax_ylim)
        ax2.set_ylim(ax_ylim)
        ax_top.set_xlim(ax_xlim)
        ax_right.set_ylim(ax_ylim)

        ax.tick_params(top=False, bottom=False, left=False, right=False)
        ax2.tick_params(top=False, bottom=False, left=False, right=False)
        ax_top.tick_params(top=False, bottom=False, left=True, right=False,
                           labeltop=False, labelbottom=False, labelleft=True, labelright=False)
        ax_right.tick_params(top=True, bottom=False, left=False, right=False,
                             labeltop=True, labelbottom=False, labelleft=False, labelright=False)

        for idx, spine in enumerate(ax_top.spines.values()):
            if idx == 0:
                continue
            spine.set_visible(False)
        for idx, spine in enumerate(ax_right.spines.values()):
            if idx == 3:
                continue
            spine.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        
        ax_empty1.axis('off')
        ax_empty2.axis('off')
        ax_legend.axis('off')

        #plt.tight_layout()
        
        return f, (ax, ax2, ax_top, ax_right, ax_legend)
        