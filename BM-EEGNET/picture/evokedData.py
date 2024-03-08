import os
import numpy as np
import mne
from matplotlib import pyplot as plt

# 如果没有数据则用这个自动下载
# sample_data_folder = mne.datasets.sample.data_path()
# 已有数据，则直接加载即可
sample_data_folder = mne.datasets.sample.data_path()
sample_data_evk_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis-ave.fif')
evokeds_list = mne.read_evokeds(sample_data_evk_file, baseline=(None, 0),
                                proj=True, verbose=False)

# Show the condition names, and reassure ourselves that baseline correction has been applied.
for e in evokeds_list:
    print(f'Condition: {e.comment}, baseline: {e.baseline}')

# convert that list of Evoked objects into a dictionary
conds = ('aud/left', 'aud/right', 'vis/left', 'vis/right')
evks = dict(zip(conds, evokeds_list))
#      ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ this is equivalent to:
# {'aud/left': evokeds_list[0], 'aud/right': evokeds_list[1],
#  'vis/left': evokeds_list[2], 'vis/right': evokeds_list[3]}

# signal trace
evks['aud/left'].plot(exclude=[])
plt.show()

# pick spatial_color gfp
evks['aud/left'].plot(picks='mag', spatial_colors=True, gfp=True)
plt.show()

# 头皮地形图
times = np.linspace(0.05, 0.13, 5)
evks['aud/left'].plot_topomap(ch_type='mag', times=times, colorbar=True)
plt.show()

# 指定time = 0.09
fig = evks['aud/left'].plot_topomap(ch_type='mag', times=0.09, average=0.1)
fig.text(0.5, 0.05, 'average from 40-140 ms', ha='center')
plt.show()

# 箭头地图
mags = evks['aud/left'].copy().pick_types(meg='mag')
mne.viz.plot_arrowmap(mags.data[:, 175], mags.info, extrapolate='local')
plt.show()

# 联合图
evks['vis/right'].plot_joint()
plt.show()

# 对比图
def custom_func(x):
    return x.max(axis=1)
for combine in ('mean', 'median', 'gfp', custom_func):
    mne.viz.plot_compare_evokeds(evks, picks='eeg', combine=combine)
plt.show()

# 定制对比图
mne.viz.plot_compare_evokeds(evks, picks='MEG 1811', colors=dict(aud=0, vis=1),
                             linestyles=dict(left='solid', right='dashed'))
plt.show()

# 传入元组或列表
temp_list = list()
for idx, _comment in enumerate(('foo', 'foo', '', None, 'bar'), start=1):
    _evk = evokeds_list[0].copy()
    _evk.comment = _comment
    _evk.data *= idx  # so we can tell the traces apart
    temp_list.append(_evk)
mne.viz.plot_compare_evokeds(temp_list, picks='mag')
plt.show()

# image plot
evks['vis/right'].plot_image(picks='meg')
plt.show()

# 次地形图
mne.viz.plot_compare_evokeds(evks, picks='eeg', colors=dict(aud=0, vis=1),
                             linestyles=dict(left='solid', right='dashed'),
                             axes='topo', styles=dict(aud=dict(linewidth=1),
                                                      vis=dict(linewidth=1)))
plt.show()


