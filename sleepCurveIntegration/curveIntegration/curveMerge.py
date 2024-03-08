import os
import numpy as np
import mne
import pywt
import matplotlib.pyplot as plt
import scipy
import samplerate
from curveIntegration.extract_bispectrum import polycoherence, plot_polycoherence

data_path = '/Users/tangbo/Downloads/sleepCurveIntegration/data/sleepedf/sleep-cassette/SC4001E0-PSG.edf'
annot_path = '/Users/tangbo/Downloads/sleepCurveIntegration/data/sleepedf/sleep-cassette/SC4001EC-Hypnogram.edf'

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

raw_train = mne.io.read_raw_edf(data_path, stim_channel='marker',
                                misc=['rectal'])
annot_train = mne.read_annotations(annot_path)

raw_train.set_annotations(annot_train, emit_warning=False)

raw_train.plot(start=60, duration=60,
               scalings=dict(eeg=1e-4, resp=1e3, eog=1e-4, emg=1e-7,
                             misc=1e-1))


annotation_desc_2_event_id = {'Sleep stage W': 1,
                              'Sleep stage 1': 2,
                              'Sleep stage 2': 3,
                              'Sleep stage 3': 4,
                              'Sleep stage 4': 4,
                              'Sleep stage R': 5}

# keep last 30-min wake events before sleep and first 30-min wake events after
# sleep and redefine annotations on raw data
annot_train.crop(annot_train[1]['onset'] - 30 * 60,
                 annot_train[-2]['onset'] + 30 * 60)
raw_train.set_annotations(annot_train, emit_warning=False)

events_train, _ = mne.events_from_annotations(
    raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)

# create a new event_id that unifies stages 3 and 4
event_id = {'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5}

# plot events
fig = mne.viz.plot_events(events_train, event_id=event_id,
                          sfreq=raw_train.info['sfreq'],
                          first_samp=events_train[0, 0]).show()

# keep the color-code for further plotting
stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included

epochs = mne.Epochs(raw=raw_train, events=events_train,
                    event_id=event_id, tmin=0., tmax=tmax, baseline=None, preload=True)
epochs.plot_image(combine='gfp', sigma=2., cmap="YlGnBu_r")



# 需要分析的频带及其范围
bandFreqs = [
    {'name': 'Delta', 'fmin': 1, 'fmax': 3},
    {'name': 'Theta', 'fmin': 4, 'fmax': 7},
    {'name': 'Alpha', 'fmin': 8, 'fmax': 13},
    {'name': 'Beta', 'fmin': 14, 'fmax': 31},
    {'name': 'Gamma', 'fmin': 31, 'fmax': 40}
]

Delta = [[]]
Theta = [[]]
Alpha = [[]]
Beta = [[]]
Gamma = [[]]
DeltaNormalized = []
ThetaNormalized = []
AlphaNormalized = []
BetaNormalized = []
BetaNormalized = []

# 绘图验证结果
plt.figure(figsize=(15, 10))
# 获取采样频率
sfreq = epochs.info['sfreq']
# 想要分析的目标频带
bandIndex = 5
# 想要分析的channel
channelIndex = 0
# 想要分析的epoch
epochIndex = 0


def plot_signal(audio_data, title=None):
    plt.figure(figsize=(9, 3.0), dpi=300)
    plt.plot(audio_data, linewidth=1)
    plt.title(title, fontsize=16)
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.show()


# 归一化
def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min);
    return x;


# 绘制归一化图例和图名
def plot_Normalization(x, name):
    plt.figure(figsize=(15, 10))
    plt.plot(Delta[0], x, c=(1, 0, 0), label='normalizationLine')
    plt.legend()
    plt.title(name)
    plt.show()


# 绘制原始数据
for i in range(0, bandIndex):
    plt.plot(epochs.get_data()[i][channelIndex], label='Raw')
    # 计算FIR滤波后的数据并绘图（注意这里要使用copy方法，否则会改变原始数据）
    firFilter = epochs.copy().filter(bandFreqs[i]['fmin'], bandFreqs[i]['fmax'])
    plt.plot(firFilter.get_data()[i][channelIndex], c=(1, 0, 0), label='FIR_Filter')

    # 绘制图例和图名
    plt.legend()
    plt.title(bandFreqs[i]['name'])
    plt.show()
    ####################################FFT对比两种方法的频谱分布
    plt.figure(figsize=(15, 10))
    # 对FIR滤波后的数据进行FFT变换
    firData = firFilter.get_data()[epochIndex][channelIndex]
    mneFIRFreq = np.abs(scipy.fft(firData))

    # 想要绘制的点数
    pointPlot = 500
    # FIR滤波后x轴对应的频率幅值范围
    FIR_X = np.linspace(0, sfreq / 2, int(mneFIRFreq.shape[0] / 2))
    # 绘制FIR滤波后的频谱分布
    plt.plot(FIR_X[:pointPlot], mneFIRFreq[:pointPlot], c=(1, 0, 0), label='FIR_Filter_fre')

    # 绘制图例和图名
    plt.legend()
    plt.title(bandFreqs[i]['name'])
    plt.show()

    freq1, freq2, bi_spectrum = polycoherence(firData, nfft=1024, nperseg=256, noverlap=100, fs=1000,
                                              norm=None)
    bi_spectrum = np.array(abs(bi_spectrum))  # calculate bi_spectrum
    bi_spectrum = 255 * (bi_spectrum - np.min(bi_spectrum)) / (np.max(bi_spectrum) - np.min(bi_spectrum))
    plot_polycoherence(freq1, freq2, bi_spectrum)

    if i == 0:
        Delta = np.array([FIR_X[:pointPlot], mneFIRFreq[:pointPlot]])
        DeltaNormalized = MaxMinNormalization(Delta[1], np.max(Delta[1]), np.min(Delta[1]))
        plot_Normalization(DeltaNormalized, "DeltaNormalized")
    elif i == 1:
        Theta = np.array([FIR_X[:pointPlot], mneFIRFreq[:pointPlot]])
        ThetaNormalized = MaxMinNormalization(Theta[1], np.max(Theta[1]), np.min(Theta[1]))
        plot_Normalization(ThetaNormalized, "ThetaNormalized")
    elif i == 2:
        Alpha = np.array([FIR_X[:pointPlot], mneFIRFreq[:pointPlot]])
        AlphaNormalized = MaxMinNormalization(Alpha[1], np.max(Alpha[1]), np.min(Alpha[1]))
        plot_Normalization(AlphaNormalized, "AlphaNormalized")
    elif i == 3:
        Beta = np.array([FIR_X[:pointPlot], mneFIRFreq[:pointPlot]])
        BetaNormalized = MaxMinNormalization(Beta[1], np.max(Beta[1]), np.min(Beta[1]))
        plot_Normalization(BetaNormalized, "BetaNormalized")
    elif i == 4:
        Gamma = np.array([FIR_X[:pointPlot], mneFIRFreq[:pointPlot]])
        GammaNormalized = MaxMinNormalization(Gamma[1], np.max(Gamma[1]), np.min(Gamma[1]))
        plot_Normalization(GammaNormalized, "GammaNormalized")

mergeX = Delta[0]


def merge(x, normalizetion, k, max):
    avg = np.average(x[1])
    mergeY = max * (1 + k * (normalizetion - avg))
    return mergeY


DeltaY = merge(Delta, DeltaNormalized, 0.5, np.max(Delta[0]))
BetaY = merge(Beta, BetaNormalized, 0.2, DeltaY)
ThetaY = merge(Theta, ThetaNormalized, 0.1, BetaY)
AlphaY = merge(Alpha, AlphaNormalized, 0.1, ThetaY)
mergeY = merge(Gamma, GammaNormalized, 0.1, AlphaY)

# 绘制融合曲线图例和图名
plt.figure(figsize=(15, 10))
plt.plot(mergeX, mergeY, c=(1, 0, 0))
plt.legend()
plt.title("融合曲线")
plt.show()
