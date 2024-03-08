import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.time_frequency import tfr_morlet
from mne.stats import f_threshold_mway_rm, f_mway_rm, fdr_correction
from mne.datasets import sample

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
tmin, tmax = -0.2, 0.5


raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

include = []
raw.info['bads'] += ['MEG 2443']


picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                       stim=False, include=include, exclude='bads')

ch_name = 'MEG 1332'


reject = dict(grad=4000e-13, eog=150e-6)
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=reject)
epochs.pick_channels([ch_name])

epochs.equalize_event_counts(event_id)


decim = 2
freqs = np.arange(7, 30, 3)
n_cycles = freqs / freqs[0]
zero_mean = False

epochs_power = list()
for condition in [epochs[k] for k in event_id]:
    this_tfr = tfr_morlet(condition, freqs, n_cycles=n_cycles,
                          decim=decim, average=False, zero_mean=zero_mean,
                          return_itc=False)
    this_tfr.apply_baseline(mode='ratio', baseline=(None, 0))
    this_power = this_tfr.data[:, 0, :, :]  # we only have one channel.
    epochs_power.append(this_power)

n_conditions = len(epochs.event_id)
n_replications = epochs.events.shape[0] // n_conditions

factor_levels = [2, 2]
effects = 'A*B'
n_freqs = len(freqs)
times = 1e3 * epochs.times[::decim]
n_times = len(times)

data = np.swapaxes(np.asarray(epochs_power), 1, 0)

data = data.reshape(n_replications, n_conditions, n_freqs * n_times)

fvals, pvals = f_mway_rm(data, factor_levels, effects=effects)

effect_labels = ['modality', 'location', 'modality by location']

# let's visualize our effects by computing f-images
for effect, sig, effect_label in zip(fvals, pvals, effect_labels):
    plt.figure()
    # show naive F-values in gray
    plt.imshow(effect.reshape(8, 211), cmap='Blues_r', extent=[times[0],
               times[-1], freqs[0], freqs[-1]], aspect='auto',
               origin='lower')
    # create mask for significant Time-frequency locations
    effect[sig >= 0.05] = np.nan
    plt.imshow(effect.reshape(8, 211), cmap='RdBu_r', extent=[times[0],
               times[-1], freqs[0], freqs[-1]], aspect='auto',
               origin='lower')
    plt.colorbar()
    plt.xlabel('时间 (ms)')
    plt.ylabel('频率 (Hz)')
    plt.show()