import mne
import matplotlib.pyplot as plt


data_path = '/Users/tangbo/Downloads/sleepCurveIntegration/data/sleepedf/sleep-cassette/SC4001E0-PSG.edf'
annot_path = '/Users/tangbo/Downloads/sleepCurveIntegration/data/sleepedf/sleep-cassette/SC4001EC-Hypnogram.edf'

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

raw_train = mne.io.read_raw_edf(data_path, stim_channel='marker',
                                misc=['rectal'])

annot_train = mne.read_annotations(annot_path)

raw_train.set_annotations(annot_train, emit_warning=False)

events_from_annot, event_dict = mne.events_from_annotations(raw_train)


custom_mapping = {'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 4, 'Sleep stage ?': 5, 'Sleep stage R': 6, 'Sleep stage W': 7}
(events_from_annot,
 event_dict) = mne.events_from_annotations(raw_train, event_id=custom_mapping)

epochs = mne.Epochs(raw_train, events=events_from_annot,
                    event_id=event_dict)

epochs.plot_image(cmap="YlGnBu_r")

sfreq=raw_train.info['sfreq']

data,times=raw_train[:3,int(sfreq*1):int(sfreq*3)]
plt.plot(times,data[0],label='Fpz_Cz',color='blue', ms=5, alpha=0.7,linewidth='2')
plt.plot(times,data[1],label='Pz_Oz',color='green', ms=5, alpha=0.7,linewidth='2')
plt.plot(times,data[2],label='ECG',color='orange', ms=5, alpha=0.7,linewidth='2')
plt.title("信号通道采样图")
plt.legend(loc='upper left', frameon=True)
plt.show()