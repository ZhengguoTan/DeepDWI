import os

import numpy as np
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.realpath(__file__))

train_loss = [
    0.980421 ,
    0.928528 ,
    0.908880 ,
    0.898629 ,
    0.891964 ,
    0.887132 ,
    0.883358 ,
    0.880341 ,
    0.877881 ,
    0.875829 ,
    0.874043 ,
    0.872525 ,
    0.871204 ,
    0.870034 ,
    0.868982 ,
    0.868019 ,
    0.867132 ,
    0.866317 ,
    0.865571 ,
    0.864880 ,
    0.864182 ,
    0.863587 ,
    0.863063 ,
    0.862592 ,
    0.862063 ,
    0.861541 ,
    0.861167 ,
    0.860724 ,
    0.860351 ,
    0.859977 ,
    0.859806 ,
    0.859759 ,
    0.859714 ,
    0.859675 ,
    0.859637 ,
    0.859599 ,
    0.859560 ,
    0.859521 ,
    0.859482 ,
    0.859443 ,
    0.859404 ,
    0.859364 ,
    0.859324 ,
    0.859284 ,
    0.859244 ,
    0.859204 ,
    0.859163 ,
    0.859123 ,
    0.859082 ,
    0.859042 ,
    0.859001 ,
    0.858960 ,
    0.858920 ,
    0.858878 ,
    0.858837 ,
    0.858796 ,
    0.858754 ,
    0.858713 ,
    0.858671 ,
    0.858629 ,
    0.858603 ,
    0.858599 ,
    0.858593 ,
    0.858589 ,
    0.858584 ,
    0.858580 ,
    0.858575 ,
    0.858571 ,
    0.858566 ,
    0.858562 ,
    0.858557 ,
    0.858552 ,
    0.858548 ,
    0.858543 ,
    0.858538 ,
    0.858534 ,
    0.858529 ,
    0.858524 ,
    0.858519 ,
    0.858514 ,
    0.858509 ,
    0.858504 ,
    0.858499 ,
    0.858494 ,
    0.858489 ,
    0.858484 ,
    0.858479 ,
    0.858473 ,
    0.858468 ,
    0.858463 ,
    0.858461 ,
    0.858461 ,
    0.858461 ,
    0.858460 ,
    0.858460 ,
    0.858459 ,
    0.858458 ,
    0.858458 ,
    0.858457 ,
    0.858457
]

valid_loss = [
    0.941501 ,
    0.904021 ,
    0.890391 ,
    0.883432 ,
    0.878945 ,
    0.875712 ,
    0.873204 ,
    0.871184 ,
    0.869547 ,
    0.868177 ,
    0.867018 ,
    0.866011 ,
    0.865140 ,
    0.864361 ,
    0.863658 ,
    0.863022 ,
    0.862440 ,
    0.861908 ,
    0.861406 ,
    0.860935 ,
    0.860517 ,
    0.860111 ,
    0.859778 ,
    0.859457 ,
    0.859142 ,
    0.858850 ,
    0.858595 ,
    0.858340 ,
    0.858081 ,
    0.857854 ,
    0.857775 ,
    0.857749 ,
    0.857726 ,
    0.857703 ,
    0.857681 ,
    0.857659 ,
    0.857637 ,
    0.857614 ,
    0.857592 ,
    0.857569 ,
    0.857547 ,
    0.857524 ,
    0.857501 ,
    0.857478 ,
    0.857455 ,
    0.857432 ,
    0.857409 ,
    0.857386 ,
    0.857362 ,
    0.857339 ,
    0.857315 ,
    0.857292 ,
    0.857268 ,
    0.857244 ,
    0.857221 ,
    0.857196 ,
    0.857172 ,
    0.857148 ,
    0.857124 ,
    0.857100 ,
    0.857089 ,
    0.857090 ,
    0.857087 ,
    0.857084 ,
    0.857082 ,
    0.857079 ,
    0.857076 ,
    0.857074 ,
    0.857072 ,
    0.857069 ,
    0.857066 ,
    0.857064 ,
    0.857061 ,
    0.857058 ,
    0.857056 ,
    0.857053 ,
    0.857050 ,
    0.857048 ,
    0.857045 ,
    0.857042 ,
    0.857039 ,
    0.857037 ,
    0.857034 ,
    0.857031 ,
    0.857028 ,
    0.857025 ,
    0.857022 ,
    0.857019 ,
    0.857016 ,
    0.857013 ,
    0.857013 ,
    0.857013 ,
    0.857013 ,
    0.857013 ,
    0.857013 ,
    0.857012 ,
    0.857012 ,
    0.857012 ,
    0.857011 ,
    0.857011
]

lamda = [
    0.044378,
    0.041061,
    0.038818,
    0.037133,
    0.035726,
    0.034479,
    0.033341,
    0.032292,
    0.031316,
    0.030408,
    0.029564,
    0.028777,
    0.028042,
    0.027356,
    0.026714,
    0.026115,
    0.025554,
    0.025029,
    0.024535,
    0.024071,
    0.023635,
    0.023224,
    0.022840,
    0.022478,
    0.022138,
    0.021822,
    0.021524,
    0.021245,
    0.020988,
    0.020750,
    0.020728,
    0.020707,
    0.020686,
    0.020664,
    0.020643,
    0.020621,
    0.020599,
    0.020576,
    0.020554,
    0.020532,
    0.020509,
    0.020487,
    0.020464,
    0.020441,
    0.020418,
    0.020395,
    0.020372,
    0.020349,
    0.020326,
    0.020303,
    0.020280,
    0.020257,
    0.020235,
    0.020212,
    0.020189,
    0.020166,
    0.020143,
    0.020120,
    0.020098,
    0.020075,
    0.020073,
    0.020071,
    0.020068,
    0.020066,
    0.020064,
    0.020061,
    0.020059,
    0.020057,
    0.020054,
    0.020052,
    0.020049,
    0.020047,
    0.020044,
    0.020042,
    0.020039,
    0.020037,
    0.020034,
    0.020031,
    0.020029,
    0.020026,
    0.020023,
    0.020021,
    0.020018,
    0.020015,
    0.020012,
    0.020010,
    0.020007,
    0.020004,
    0.020001,
    0.019998,
    0.019998,
    0.019998,
    0.019997,
    0.019997,
    0.019997,
    0.019996,
    0.019996,
    0.019996,
    0.019995,
    0.019995
]

fontsize = 14

epochs = range(1, len(train_loss)+1, 1)

fig, ax1 = plt.subplots(figsize=(6,3))

color = 'tab:red'
ax1.set_xlabel('epochs', fontsize=fontsize)
# ax1.set_ylabel('loss', color=color)
ln1 = ax1.plot(epochs, train_loss, '-', color=color, label='training')
ln2 = ax1.plot(epochs, valid_loss, '--', color=color, label='validation')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
# ax2.set_ylabel('$\lambda$', color=color)  # we already handled the x-label with ax1
ln3 = ax2.plot(epochs, lamda, color=color, label='$\lambda$')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0.01, 0.05])

lns = ln1 + ln2 + ln3
lbs = [l.get_label() for l in lns]
ax1.legend(lns, lbs, loc=1, fontsize=fontsize)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.xlim([1, 100])
plt.title('ADMM Unrolling Convergence')

plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(DIR + '/convergence.png',
            bbox_inches='tight', pad_inches=0, dpi=300)

plt.show()