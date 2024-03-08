import os.path as op

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import (make_axes_locatable, ImageGrid,
                                     inset_locator)

import mne

plt.rcParams['font.sans-serif'] = ['SimHei']

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
fname_stc = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-eeg-lh.stc')
fname_evoked = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')

evoked = mne.read_evokeds(fname_evoked, 'Left Auditory')
evoked.pick_types(meg='grad').apply_baseline((None, 0.))
max_t = evoked.get_peak()[1]

stc = mne.read_source_estimate(fname_stc)

stc.plot(views='lat', hemi='split', size=(800, 400), subject='sample',
         subjects_dir=subjects_dir, initial_time=max_t,
         time_viewer=False, show_traces=False)
colormap = 'viridis'
clim = dict(kind='value', lims=[4, 8, 12])

# Plot the STC, get the brain image, crop it:
brain = stc.plot(views='lat', hemi='split', size=(800, 400), subject='sample',
                 subjects_dir=subjects_dir, initial_time=max_t, background='w',
                 colorbar=False, clim=clim, colormap=colormap,
                 time_viewer=False, show_traces=False)
screenshot = brain.screenshot()
brain.close()
nonwhite_pix = (screenshot != 255).any(-1)
nonwhite_row = nonwhite_pix.any(1)
nonwhite_col = nonwhite_pix.any(0)
cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

# before/after results
fig = plt.figure(figsize=(4, 4))
axes = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.5)
print(axes)
for ax, image in zip(axes, [screenshot, cropped_screenshot]):
    ax.imshow(image)
plt.show()
