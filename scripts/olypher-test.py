#!/usr/bin/env python

from pylab import *
import scanr.tracking
from scanr.session import SessionData
from ..tools.stats import KT_estimate
from ..tools.filters import circular_blur
import numpy as np

reload(scanr.tracking)
from scanr.tracking import SpatialInformationScore

data = SessionData(rds=(97,6,2))
cluster = data.cluster_data((1,2))

score = SpatialInformationScore(measure='olypher', x_bins=48, reverse_for_shuffle=True,
    random=False, fixed_spacing=0.5, min_offset=30.0)

H_xk = score.H_xk(data, cluster)
P_k = KT_estimate(H_xk.sum(axis=0))
P_k_x = (H_xk.astype('d') + 0.5) / (
    H_xk.sum(axis=1)[:,np.newaxis] + 0.5*H_xk.shape[1])
I_pos = np.sum(P_k_x * np.log2(P_k_x / P_k), axis=1)
I = circular_blur(I_pos, 360./score.x_bins)

print I.max()

figure(1)
clf()
subplot(411)
cla()
imshow(P_k_x.T, origin='lower', interpolation='nearest', aspect='auto')
colorbar()
ylabel("P_k|x")

subplot(412)
cla()
D1 = np.log2(P_k_x / P_k)
imshow(D1.T, origin='lower', interpolation='nearest', aspect='auto')
colorbar()
ylabel("log(P_k|x / P_k)")

subplot(413)
cla()
D2 = P_k_x * np.log2(P_k_x / P_k)
imshow(D2.T, origin='lower', interpolation='nearest', aspect='auto')
colorbar()
ylabel("P_k|x * log(P_k|x / P_k)")

subplot(414)
cla()
bins = np.linspace(0, 360, score.x_bins)
plot(bins, I_pos, 'r-')
plot(bins, I, 'k-')
xlim(0, 360)
ylabel("I_pos")
h = colorbar()
h.ax.set_visible(False)
draw()

print '--'

I = score.compute(data, cluster)
p = score.pval(data, cluster)
print '%.3f bits,'%I, 'p < %.4f'%p

