import numpy as np
import matplotlib.pyplot as plt
from submm.utils import plotting

# python script to demonstrate some extra interactive plotting tools
# I like to have activated

# initialize figure and axes class
fig = plt.figure()
ax = fig.add_subplot(111)

# plot something
ax.plot(np.linspace(0, 100, 100), np.sin(np.linspace(0, 4 * np.pi, 100)))

# call the interactive plotting interface
ip = plotting.activate_interactive_plotting(fig, ax)

print("Data points grabbed")
for i in range(0, len(ip.x_data)):
    print(ip.x_data[i], ip.y_data[i])
