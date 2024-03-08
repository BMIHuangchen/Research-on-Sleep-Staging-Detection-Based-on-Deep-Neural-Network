import os
import numpy as np
import mne
import pywt
import matplotlib.pyplot as plt
import scipy
import samplerate
from curveIntegration.extract_bispectrum import polycoherence, plot_polycoherence

data_path = 'D:/project_python/data/SC4621E0-PSG.edf'
annot_path = 'D:/project_python/data/SC4621EV-Hypnogram.edf'


npz = np.load("D:\project_python\sleepCurveIntegration\data\sleepedf\sleep-cassette\eeg_fpz_cz\SC4001E0.npz",allow_pickle=True)
#预测
x=npz['x']
y=npz['y']
fs=npz['fs']
ch_label =npz['ch_label']
start_datetime=npz['start_datetime']
file_duration =npz ['file_duration']
epoch_duration =npz['epoch_duration']
n_all_epochs =npz['n_all_epochs']
n_epochs =npz['n_epochs']
print(len(x))
print(len(y))
print(fs)
print(ch_label)
print(start_datetime)
print(file_duration)
print(epoch_duration)
print(n_all_epochs)
print(n_epochs)
# raw_train = mne.io.read_raw_edf(data_path, stim_channel='marker',
#                                 misc=['rectal'])
#
# annot_train = mne.read_annotations(annot_path)
#
# print(raw_train)
# raw_train.set_annotations(annot_train, emit_warning=False)
#
# events_from_annot, event_dict = mne.events_from_annotations(raw_train)
#
#
# custom_mapping = {'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 4, 'Sleep stage ?': 5, 'Sleep stage R': 6, 'Sleep stage W': 7}
# (events_from_annot,
#  event_dict) = mne.events_from_annotations(raw_train, event_id=custom_mapping)
#
# epochs = mne.Epochs(raw_train, events=events_from_annot,
#                     event_id=event_dict)
#
# epochs.plot_image(cmap="YlGnBu_r")
#
# sfreq=raw_train.info['sfreq']
#
# data,times=raw_train[:3,int(sfreq*1):int(sfreq*3)]
# plt.plot(times,data.T)
# plt.title("Sample channels")
# plt.show()
#
# raw_train.plot(start=60, duration=60,
#                scalings=dict(eeg=1e-4, resp=1e3, eog=1e-4, emg=1e-7,
#                              misc=1e-1))
#
#
# annotation_desc_2_event_id = {'Sleep stage W': 1,
#                               'Sleep stage 1': 2,
#                               'Sleep stage 2': 3,
#                               'Sleep stage 3': 4,
#                               'Sleep stage 4': 4,
#                               'Sleep stage R': 5}
#
# # keep last 30-min wake events before sleep and first 30-min wake events after
# # sleep and redefine annotations on raw data
# annot_train.crop(annot_train[1]['onset'] - 30 * 60,
#                  annot_train[-2]['onset'] + 30 * 60)
# raw_train.set_annotations(annot_train, emit_warning=False)
#
# events_train, _ = mne.events_from_annotations(
#     raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)
#
# # create a new event_id that unifies stages 3 and 4
# event_id = {'Sleep stage W': 1,
#             'Sleep stage 1': 2,
#             'Sleep stage 2': 3,
#             'Sleep stage 3/4': 4,
#             'Sleep stage R': 5}
#
# # plot events
# # fig = mne.viz.plot_events(events_train, event_id=event_id,
# #                           sfreq=raw_train.info['sfreq'],
# #                           first_samp=events_train[0, 0])
#
#
#
# tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included
#
# epochs = mne.Epochs(raw=raw_train, events=events_train,
#                     event_id=event_id, tmin=0., tmax=tmax, baseline=None, preload=True)
#
# # epochs.plot_image(combine='gfp', sigma=2., cmap="YlGnBu_r")



