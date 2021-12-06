#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, axs = plt.subplots(3, 2)

axs[0, 0].plot(y0, color='red')
axs[0, 0].axis(xmin=0, xmax=10)

axs[0, 1].scatter(x1, y1, s=8, color='magenta')
axs[0, 1].set_title('Men\'s Height vs Weight', size='x-small')
axs[0, 1].set_xlabel('Height (in)', size='x-small')
axs[0, 1].set_ylabel('Weight (lbs)', size='x-small')

axs[1, 0].plot(x2, y2)
axs[1, 0].axis(xmin=0, xmax=28650)
axs[1, 0].set_yscale('log')
axs[1, 0].set_title('Exponential Decay of C-14', size='x-small')
axs[1, 0].set_xlabel('Time (years)', size='x-small')
axs[1, 0].set_ylabel('Fraction Remaining', size='x-small')

axs[1, 1].plot(x3, y31, 'r--', label='C-14')
axs[1, 1].plot(x3, y32, 'g', label='Ra-226')
axs[1, 1].axis(xmin=0, xmax=20000, ymin=0, ymax=1)
axs[1, 1].set_title('Exponential Decay of Radioactive Elements',
                    size='x-small')
axs[1, 1].set_xlabel('Time (years)', size='x-small')
axs[1, 1].set_ylabel('Fraction Remaining', size='x-small')
axs[1, 1].legend(fontsize='x-small')

gs = axs[2, 0].get_gridspec()
for ax in axs[2, :]:
    ax.remove()
axbig = fig.add_subplot(gs[2, :])

x4 = np.arange(0, 110, 10)
axbig.hist(student_grades, bins=x4, edgecolor='black')
axbig.set_xticks(x4)
axbig.axis(xmin=0, xmax=100, ymin=0, ymax=30)
axbig.set_title('Project A', size='x-small')
axbig.set_xlabel('Grades', size='x-small')
axbig.set_ylabel('Number of Students', size='x-small')

fig.suptitle('All in One')
plt.tight_layout()
plt.show()
