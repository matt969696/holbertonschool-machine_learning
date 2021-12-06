#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
b1 = fruit[0]
b2 = fruit[0] + fruit[1]
b3 = fruit[0] + fruit[1] + fruit[2]

cols = ('Farrah', 'Fred', 'Felicia')
rows = ('apples', 'bananas', 'oranges', 'peaches')

fig, ax = plt.subplots()
ax.bar(cols, fruit[0], color='red', label='apples', width=0.5)
ax.bar(cols, fruit[1], color='yellow', label='bananas', bottom=b1, width=0.5)
ax.bar(cols, fruit[2], color='#ff8000', label='oranges', bottom=b2, width=0.5)
ax.bar(cols, fruit[3], color='#ffe5b4', label='peaches', bottom=b3, width=0.5)

plt.yticks(np.arange(0, 90, 10))
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.legend()

plt.show()
