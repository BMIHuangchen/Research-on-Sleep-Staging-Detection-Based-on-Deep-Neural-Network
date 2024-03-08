import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import (read_inverse_operator,
                              make_inverse_resolution_matrix,
                              get_point_spread)

from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle


data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-fixed-inv.fif'
forward = mne.read_forward_solution(fname_fwd)
# Convert forward solution to fixed source orientations
mne.convert_forward_solution(
    forward, surf_ori=True, force_fixed=True, copy=False)
inverse_operator = read_inverse_operator(fname_inv)

# Compute resolution matrices for MNE
rm_mne = make_inverse_resolution_matrix(forward, inverse_operator,
                                        method='MNE', lambda2=1. / 3.**2)
src = inverse_operator['src']
del forward, inverse_operator  # save memory

labels = mne.read_labels_from_annot('sample', parc='aparc',
                                    subjects_dir=subjects_dir)
n_labels = len(labels)
label_colors = [label.color for label in labels]
# First, we reorder the labels based on their location in the left hemi
label_names = [label.name for label in labels]
lh_labels = [name for name in label_names if name.endswith('lh')]

# Get the y-location of the label
label_ypos = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels[idx].pos[:, 1])
    label_ypos.append(ypos)

# Reorder the labels based on their location
lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

# For the right hemi
rh_labels = [label[:-2] + 'rh' for label in lh_labels]

# Compute first PCA component across PSFs within labels.
# Note the differences in explained variance, probably due to different
# spatial extents of labels.
n_comp = 5
stcs_psf_mne, pca_vars_mne = get_point_spread(
    rm_mne, src, labels, mode='pca', n_comp=n_comp, norm=None,
    return_pca_vars=True)
n_verts = rm_mne.shape[0]
del rm_mne
with np.printoptions(precision=1):
    for [name, var] in zip(label_names, pca_vars_mne):
        print(f'{name}: {var.sum():.1f}% {var}')
# get PSFs from Source Estimate objects into matrix
psfs_mat = np.zeros([n_labels, n_verts])
# Leakage matrix for MNE, get first principal component per label
for [i, s] in enumerate(stcs_psf_mne):
    psfs_mat[i, :] = s.data[:, 0]
# Compute label-to-label leakage as Pearson correlation of PSFs
# Sign of correlation is arbitrary, so take absolute values
leakage_mne = np.abs(np.corrcoef(psfs_mat))

# Save the plot order and create a circular layout
node_order = lh_labels[::-1] + rh_labels  # mirror label order across hemis
node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) / 2])
# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 200 strongest connections.
fig = plt.figure(num=None, figsize=(8, 8), facecolor='black')
plot_connectivity_circle(leakage_mne, label_names, n_lines=200,
                         node_angles=node_angles, node_colors=label_colors, fig=fig)