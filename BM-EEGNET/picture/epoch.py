import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import compute_proj_ecg
from mne_connectivity import envelope_correlation

plt.rcParams['font.sans-serif'] = ['SimHei']

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')

raw = mne.io.read_raw_fif(sample_data_raw_file)

raw.crop(tmax=150).resample(100).pick('eeg')
ecg_proj, _ = compute_proj_ecg(raw, ch_name='EEG 050')  # No ECG chan
raw.add_proj(ecg_proj)
raw.apply_proj()

epochs = mne.make_fixed_length_epochs(raw, duration=30, preload=False)

# event_related_plot = epochs.plot_image(picks=['EEG 050'])

epochs.load_data().filter(l_freq=8, h_freq=12)
alpha_data = epochs.get_data()
corr_matrix = envelope_correlation(alpha_data).get_data()
first_30 = corr_matrix[0]
last_30 = corr_matrix[-1]
corr_matrices = [first_30, last_30]
color_lims = np.percentile(np.array(corr_matrices), [5, 95])
titles = ['K-means 前 ', ' K-means 后']
y_lable ="相关系数"

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.suptitle('经过K-means算法前后的相关矩阵')
for ci, corr_matrix in enumerate(corr_matrices):
    ax = axes[ci]
    mpbl = ax.imshow(corr_matrix.squeeze(), clim=color_lims)
    ax.set_xlabel(titles[ci])
    ax.set_ylabel(y_lable)

cax = fig.add_axes([1, 0.2, 0.025, 0.6])
cbar = fig.colorbar(ax.images[0], cax=cax)
cbar.set_label('Correlation Coefficient')
plt.show()