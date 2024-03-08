import os
import numpy as np
import matplotlib.pyplot as plt
import mne

# 两点距离
def distance(e1, e2):
    return np.sqrt((e1[0]-e2[0])**2+(e1[1]-e2[1])**2)

# 集合中心
def means(arr):
    return np.array([np.mean([e[0] for e in arr]), np.mean([e[1] for e in arr])])

# arr中距离a最远的元素，用于初始化聚类中心
def farthest(k_arr, arr):
    f = [0, 0]
    max_d = 0
    for e in arr:
        d = 0
        for i in range(k_arr.__len__()):
            d = d + np.sqrt(distance(k_arr[i], e))
        if d > max_d:
            max_d = d
            f = e
    return f

# arr中距离a最近的元素，用于聚类
def closest(a, arr):
    c = arr[1]
    min_d = distance(a, arr[1])
    arr = arr[1:]
    for e in arr:
        d = distance(a, e)
        if d < min_d:
            min_d = d
            c = e
    return c

if __name__=="__main__":
    ## 生成二维随机坐标，手上有数据集的朋友注意，理解arr改起来就很容易了
    ## arr是一个数组，每个元素都是一个二元组，代表着一个坐标
    ## arr形如：[ (x1, y1), (x2, y2), (x3, y3) ... ]
    arr = np.random.randint(100, size=(100, 1, 2))[:, 0, :]

    ## 初始化聚类中心和聚类容器
    m = 5
    r = np.random.randint(arr.__len__() - 1)
    k_arr = np.array([arr[r]])
    cla_arr = [[]]
    for i in range(m-1):
        k = farthest(k_arr, arr)
        k_arr = np.concatenate([k_arr, np.array([k])])
        cla_arr.append([])

    ## 迭代聚类
    n = 20
    cla_temp = cla_arr
    for i in range(n):    # 迭代n次
        for e in arr:    # 把集合里每一个元素聚到最近的类
            ki = 0        # 假定距离第一个中心最近
            min_d = distance(e, k_arr[ki])
            for j in range(1, k_arr.__len__()):
                if distance(e, k_arr[j]) < min_d:    # 找到更近的聚类中心
                    min_d = distance(e, k_arr[j])
                    ki = j
            cla_temp[ki].append(e)
        # 迭代更新聚类中心
        for k in range(k_arr.__len__()):
            if n - 1 == i:
                break
            k_arr[k] = means(cla_temp[k])
            cla_temp[k] = []

    ## 可视化展示
    col = ['HotPink', 'Aqua', 'Chartreuse', 'yellow', 'LightSalmon']
    for i in range(m):
        plt.scatter(k_arr[i][0], k_arr[i][1], linewidth=10, color=col[i])
        plt.scatter([e[0] for e in cla_temp[i]], [e[1] for e in cla_temp[i]], color=col[i])
    plt.xticks([])
    plt.yticks([])
    plt.show()
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                        'sample_audvis_raw.fif')
    raw = mne.io.read_raw_fif(sample_data_raw_file)
    raw.crop(tmax=60)
    n_time_samps = raw.n_times
    time_secs = raw.times
    ch_names = raw.ch_names
    n_chan = len(ch_names)
    eeg_and_eog = raw.copy().pick_types(meg=False, eeg=True, eog=True)
    raw_temp = raw.copy()
    raw_temp.drop_channels(['EEG 037', 'EEG 059'])
    raw_temp.pick_channels(['MEG 1811', 'EEG 017', 'EOG 061'])
    channel_names = ['EOG 061', 'EEG 003', 'EEG 002', 'EEG 001']
    eog_and_frontal_eeg = raw.copy().reorder_channels(channel_names)
    raw.rename_channels({'EOG 061': 'blink detector'})
    channel_renaming_dict = {name: name.replace(' ', '_') for name in raw.ch_names}
    raw.rename_channels(channel_renaming_dict)
    raw.set_channel_types({'EEG_001': 'eog'})
    raw_selection = raw.copy().crop(tmin=10, tmax=12.5)
    raw_selection.crop(tmin=1)
    raw_selection1 = raw.copy().crop(tmin=30, tmax=30.1)  # 0.1 seconds
    raw_selection2 = raw.copy().crop(tmin=40, tmax=41.1)  # 1.1 seconds
    raw_selection3 = raw.copy().crop(tmin=50, tmax=51.3)  # 1.3 seconds
    raw_selection1.append([raw_selection2, raw_selection3])

    sampling_freq = raw.info['sfreq']
    start_stop_seconds = np.array([11, 13])
    start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)
    channel_index = 0
    raw_selection = raw[channel_index, start_sample:stop_sample]

    x = raw_selection[1]
    y = raw_selection[0].T
    plt.plot(x, y)
    plt.show()


