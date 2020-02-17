import numpy as np
import pandas as pd

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.gridspec as gridspec

from numbers import Number
from copy import copy
        

class OncoPrint:
    class __PatchLegendElement:
        def __init__(self, patch):
            self.patch = patch
            
            
    class __PatchLegendHandler:
        def __init__(self, bgcolor, bgsize=[1, 1]):
            self.bgcolor = bgcolor
            self.bgsize = bgsize
            
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0 = handlebox.xdescent - handlebox.width * 0.5 * (self.bgsize[0] - 1)
            y0 = handlebox.ydescent - handlebox.height * 0.5 * (self.bgsize[1] - 1)
            width, height = handlebox.width * self.bgsize[0], handlebox.height * self.bgsize[1]
            bgcolor = self.bgcolor
            bg = Rectangle([x0, y0], width, height, color=bgcolor, lw=0, transform=handlebox.get_transform())
            patch = orig_handle.patch
            patch.set_transform(Affine2D().scale(width, height).translate(x0, y0) + handlebox.get_transform())
            handlebox.add_artist(bg)
            handlebox.add_artist(patch)
            return patch
        
        
    class __ScatterLegendElement:
        def __init__(self, line2d):
            self.line2d = line2d

            
    class __ScatterLegendHandler(HandlerLine2D):
        def __init__(self, bgcolor, bgsize=[1, 1], marker_pad=0.3, numpoints=1, **kw):
            self.bgcolor = bgcolor
            self.bgsize = bgsize
            HandlerLine2D.__init__(self, marker_pad=marker_pad, numpoints=numpoints, **kw)

        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            xdescent, ydescent, width, height = self.adjust_drawing_area(
                     legend, orig_handle.line2d,
                     handlebox.xdescent, handlebox.ydescent,
                     handlebox.width, handlebox.height,
                     fontsize)
            artists = self.create_artists(legend, orig_handle.line2d,
                                          xdescent, ydescent, width, height,
                                          fontsize, handlebox.get_transform())

            x0 = handlebox.xdescent - handlebox.width * 0.5 * (self.bgsize[0] - 1)
            y0 = handlebox.ydescent - handlebox.height * 0.5 * (self.bgsize[1] - 1)
            width, height = handlebox.width * self.bgsize[0], handlebox.height * self.bgsize[1]
            bgcolor = self.bgcolor
            bg = Rectangle([x0, y0], width, height, color=bgcolor, lw=0, transform=handlebox.get_transform())

            handlebox.add_artist(bg)
            for a in artists:
                handlebox.add_artist(a)

            return artists[0]
        
    
    def __init__(self, recurrence_matrix, genes=None, samples=None, seperator=","):
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
        
        _, uniq_idx = np.unique(genes, return_index=True)
        if len(uniq_idx) != len(genes):
            dedup_mat = []
            dedup_genes = genes[np.sort(uniq_idx)]
            for g in dedup_genes:
                rows = mat[genes == g, :]
                joined_row = rows[0]
                for ridx in range(1, len(rows)):
                    for cidx in range(len(samples)):
                        if self._is_valid(joined_row[cidx]) and self._is_valid(rows[ridx][cidx]):
                            joined_row[cidx] += seperator + rows[ridx][cidx]
                        elif self._is_valid(rows[ridx][cidx]):
                            joined_row[cidx] = rows[ridx][cidx]
                dedup_mat.append(joined_row)
            self.mat = np.array(dedup_mat)
            self.genes = dedup_genes
        else:
            self.mat = mat
            self.genes = genes
            
        self.seperator = seperator
        self.samples = samples   
        
        
    def _is_valid(self, s):
        return isinstance(s, str) and len(s) > 0
    
    
    def _sort_genes_default(self):
        cntmat = np.zeros_like(self.sorted_mat, dtype=int)
        for i in range(self.sorted_mat.shape[0]):
            for j in range(self.sorted_mat.shape[1]):
                if self._is_valid(self.sorted_mat[i,j]):
                    cntmat[i,j] = len(np.unique(self.sorted_mat[i,j].split(self.seperator)))
        
        sorted_indices = np.argsort(np.sum(cntmat, axis=1))[::-1] # gene order
        self.sorted_genes = self.genes[sorted_indices]
        self.sorted_mat = self.sorted_mat[sorted_indices, :]
        
        
    def _sort_samples_default(self, mutation_types):
        mutation_to_weight = {mut: i for i, mut in enumerate(mutation_types[::-1], start=1)}
        weighted_filpped_cntmat = np.zeros_like(self.sorted_mat, dtype=int)
        for i in range(self.sorted_mat.shape[0]):
            for j in range(self.sorted_mat.shape[1]):
                if self._is_valid(self.sorted_mat[i,j]):
                    for mut in np.unique(self.sorted_mat[i,j].split(self.seperator)):
                        weighted_filpped_cntmat[self.sorted_mat.shape[0] - i - 1, j] += mutation_to_weight.get(mut, 0)
        sorted_indices = np.lexsort(weighted_filpped_cntmat)[::-1]
        self.sorted_samples = self.samples[sorted_indices]
        self.sorted_mat = self.sorted_mat[:, sorted_indices]
        
        
    def oncoprint(self, markers, title="", gene_sort_method='default', sample_sort_method='default', figsize=[50, 20], width_ratios=[10, 1, 1], height_ratios=[1, 5], cell_background="#dddddd", gap=0.3, ratio_template="{0:.0%}", legend_handle_size_ratio=[1, 1], legend_kwargs={}):
        f, ((ax_top, ax_empty1, ax_empty2), (ax, ax_right, ax_legend)) = plt.subplots(2, 3, figsize=figsize, gridspec_kw={'width_ratios': width_ratios, 'height_ratios': height_ratios})
                
        mutation_types = [b[0] for b in sorted(markers.items(), key=lambda a: a[1].get('zindex', 1))]
        self.sorted_mat = self.mat
        self.sorted_genes = self.genes
        self.sorted_samples = self.samples
        if gene_sort_method != 'unsorted':
            if gene_sort_method == 'default':
                self._sort_genes_default()
            else:
                print("Warning: gene sorting method '%s' is not supported."%gene_sort_method)
        if sample_sort_method != 'unsorted':
            if sample_sort_method == 'default':
                self._sort_samples_default(mutation_types)
            else:
                print("Warning: sample sorting method '%s' is not supported."%sample_sort_method)

        if isinstance(gap, Number):
            gap = np.array([gap, gap])
        else:
            assert len(gap) == 2, "The length of 'gap' is only allowed to be 2."
            gap = np.array(gap)
            
        backgrounds = []
        background_lengths = 1.0 - gap
        t_scale = Affine2D().scale(*background_lengths)
        patch_mutations = defaultdict(lambda: [[], []])
        scatter_mutations = defaultdict(lambda: [[], []])
        stacked_counts_top = np.zeros([len(mutation_types), self.sorted_mat.shape[1]])
        stacked_counts_right = np.zeros([len(mutation_types), self.sorted_mat.shape[0]])
        counts_left = np.zeros(self.sorted_mat.shape[0])
        for i in range(self.sorted_mat.shape[0]):
            for j in range(self.sorted_mat.shape[1]):
                rect = Rectangle(-background_lengths / 2.0 + (j, i, ), *background_lengths)
                backgrounds.append(rect)
                if self._is_valid(self.sorted_mat[i,j]):
                    counts_left[i] += 1
                    for mut in self.sorted_mat[i,j].split(self.seperator):
                        stacked_counts_top[mutation_types.index(mut), j] += 1
                        stacked_counts_right[mutation_types.index(mut), i] += 1
                        ms = markers[mut]
                        if isinstance(ms['marker'], str) and (ms['marker'] == 'fill' or ms['marker'] == 'rect'):
                            patch_mutations[mut][0].append(Rectangle((0, 0), 1, 1))
                            patch_mutations[mut][1].append((j, i, ))
                        elif isinstance(ms['marker'], Patch):
                            patch_mutations[mut][0].append(copy(ms['marker']))
                            patch_mutations[mut][1].append((j, i, ))
                        else:
                            scatter_mutations[mut][0].append(j)
                            scatter_mutations[mut][1].append(i)
                
        pc = PatchCollection(backgrounds, color=cell_background, linewidth=0)
        ax.add_collection(pc)
        
        legend_elements = []
        for mut in mutation_types:
            ms = markers[mut]
            mk = ms['marker']
            col = ms['color']
            if mut in patch_mutations:
                patches, coords = patch_mutations[mut]
                w, h = background_lengths * (ms.get('width', 1.0), ms.get('height', 1.0), )
                t_scale = Affine2D().scale(w, -h).translate(0, h)
                for p, (x, y) in zip(patches, coords):
                    p.set_transform(t_scale + Affine2D().translate(x - w * 0.5, y - h * 0.5))
                pc_kwargs = {k: v for k, v in ms.items() if not k in ('marker', 'width', 'height', 'zindex', )}
                if isinstance(mk, str) and (mk == 'fill' or mk == 'rect'):
                    pc_kwargs['linewidth'] = pc_kwargs.get('linewidth', 0)
                    legend_width, legend_height = ms.get('width', 1), ms.get('height', 1)
                    legend_patch = Rectangle(((1.0 - legend_width) * 0.5, (1.0 - legend_height) * 0.5, ), legend_width, legend_height)
                else:
                    legend_patch = copy(mk)
                pc = PatchCollection(patches, **pc_kwargs)
                ax.add_collection(pc)
                legend_el = self.__PatchLegendElement(PatchCollection([legend_patch], **pc_kwargs))
            elif mut in scatter_mutations:
                scatter_kwargs = {k: v for k, v in markers[mut].items() if k != 'zindex'}
                ax.scatter(*scatter_mutations[mut], **scatter_kwargs)
                line2d = Line2D([0], [0], color='#ffffff00', markerfacecolor=col, markeredgecolor=col, markeredgewidth=2, marker=mk)
                legend_el = self.__ScatterLegendElement(line2d)
            legend_elements.append(legend_el)
            
        ax_xticks = range(len(self.sorted_samples))
        ax_yticks = range(len(self.sorted_genes))

        ax_top.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_right.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        bottom = np.zeros(self.mat.shape[1])
        for idx, cnts in enumerate(stacked_counts_top):
            col = markers[mutation_types[idx]]['color']
            ax_top.bar(ax_xticks, cnts, color=col, bottom=bottom)
            bottom += cnts
            
        left = np.zeros(self.mat.shape[0])
        for idx, cnts in enumerate(stacked_counts_right):
            col = markers[mutation_types[idx]]['color']
            ax_right.barh(ax_yticks, cnts, color=col, left=left)
            left += cnts
            
        leg = ax_legend.legend(legend_elements, mutation_types,
                               handler_map={
                                   self.__PatchLegendElement: self.__PatchLegendHandler(bgcolor=cell_background, bgsize=legend_handle_size_ratio),
                                   self.__ScatterLegendElement: self.__ScatterLegendHandler(bgcolor=cell_background, bgsize=legend_handle_size_ratio)
                               }, loc='center', **legend_kwargs)
        leg_fr = leg.get_frame()
        
        leg_fr.set_edgecolor('none')
        leg_fr.set_facecolor('none')
        leg_fr.set_linewidth(0.0)

        ax.set_xticks(ax_xticks)
        ax.set_xticklabels(self.sorted_samples)
        ax.tick_params(axis='x', rotation=90)

        ax.set_yticks(ax_yticks)
        ax.set_yticklabels([ratio_template.format(e/float(self.mat.shape[1])) for e in counts_left])

        ax2 = ax.twinx()
        ax2.set_yticks(ax_yticks)
        ax2.set_yticklabels(self.sorted_genes)
        
        ax_right.tick_params(axis='x', rotation=90)

        ax_xlim = [-background_lengths[0]/2.0, self.mat.shape[1] - 1 + background_lengths[0]/2.0]
        ax_ylim = [self.mat.shape[0] - 1 + background_lengths[1]/2.0, -background_lengths[1]/2.0]

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

        plt.tight_layout()

        if title != "":
            ttl = f.suptitle(title)
            f.subplots_adjust(top=0.85)

        return f, (ax, ax2, ax_top, ax_right, ax_legend)
        