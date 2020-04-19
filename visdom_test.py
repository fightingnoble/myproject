import visdom
import numpy as np
# vis = visdom.Visdom(server='10.15.89.41', port=30330, use_incoming_socket=False)
vis = visdom.Visdom(port=30330)
x = np.random.lognormal(0,0.05,200)
vis.histogram(x)
