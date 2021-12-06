#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

x = np.arange(0, 110, 10)

plt.hist(student_grades, bins=x, edgecolor='black')
plt.xticks(x)
plt.xlim([0, 100])
plt.ylim([0, 30])
plt.title('Project A')
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.show()
