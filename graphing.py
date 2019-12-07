# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# generate random data
x=range(1,41)
y=np.random.uniform(size=40)

# initialize the figure
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')

# stem function: first way
plt.stem(x, y)
plt.ylim(0, 1.2)
# plt.show()

#multiple plots
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('blah')
ax1.stem(x, y)
ax2.stem(x, -y)
plt.show()
