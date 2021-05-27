# Copyright Activeeon 2007-2021. All rights reserved.
import math
import numpy as np
import urllib3

from io import BytesIO
from PIL import Image
from visdom import Visdom

visdom_endpoint = variables.get("ENDPOINT_VISDOM") if variables.get("ENDPOINT_VISDOM") else results[0].__str__()
print("VISDOM_ENDPOINT: ", visdom_endpoint)

assert visdom_endpoint is not None

visdom_endpoint = visdom_endpoint.replace("http://", "")

(VISDOM_HOST, VISDOM_PORT) = visdom_endpoint.split(":")

print("Connecting to %s:%s" % (VISDOM_HOST, VISDOM_PORT))
vis = Visdom(server="http://"+VISDOM_HOST, port=int(VISDOM_PORT))
assert vis.check_connection()

# text plot
textwindow = vis.text('Hello World!')
# updatetextwindow = vis.text('Hello World! More text should be here')
# vis.text('And here it is', win=updatetextwindow, append=True)

# show ActiveEon logo
url_image = 'http://s3.eu-west-2.amazonaws.com/activeeon-public/images/logo.jpg'
http = urllib3.PoolManager()
r = http.request('GET', url_image)
image = np.asarray(Image.open(BytesIO(r.data))).astype(np.uint8)
vis_image = image.transpose((2, 0, 1)).astype(np.float64)
vis.image(vis_image,opts=dict(title='ActiveEon', caption='ActiveEon'))

# boxplot
X = np.random.rand(100, 2)
X[:, 1] += 2
vis.boxplot(X=X,opts=dict(legend=['Men', 'Women']))

# stemplot
Y = np.linspace(0, 2 * math.pi, 70)
X = np.column_stack((np.sin(Y), np.cos(Y)))
vis.stem(X=X,Y=Y,opts=dict(legend=['Sine', 'Cosine']))

# quiver plot
X = np.arange(0, 2.1, .2)
Y = np.arange(0, 2.1, .2)
X = np.broadcast_to(np.expand_dims(X, axis=1), (len(X), len(X)))
Y = np.broadcast_to(np.expand_dims(Y, axis=0), (len(Y), len(Y)))
U = np.multiply(np.cos(X), Y)
V = np.multiply(np.sin(X), Y)
vis.quiver(X=U,Y=V,opts=dict(normalize=0.9))

# pie chart
X = np.asarray([19, 26, 55])
vis.pie(X=X, opts=dict(legend=['Residential', 'Non-Residential', 'Utility']))

# mesh plot
x = [0, 0, 1, 1, 0, 0, 1, 1]
y = [0, 1, 1, 0, 0, 1, 1, 0]
z = [0, 0, 0, 0, 1, 1, 1, 1]
X = np.c_[x, y, z]
i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
Y = np.c_[i, j, k]
vis.mesh(X=X, Y=Y, opts=dict(opacity=0.5))

# contour
x = np.tile(np.arange(1, 101), (100, 1))
y = x.transpose()
X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))
vis.contour(X=X, opts=dict(colormap='Viridis'))

# surface
vis.surf(X=X, opts=dict(colormap='Hot'))

# line plots
vis.line(Y=np.random.rand(10))
Y = np.linspace(-5, 5, 100)
vis.line(Y=np.column_stack((Y * Y, np.sqrt(Y + 5))),X=np.column_stack((Y, Y)),opts=dict(markers=False))

 # heatmap
vis.heatmap(X=np.outer(np.arange(1, 6), np.arange(1, 11)),
    opts=dict(
        columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        rownames=['y1', 'y2', 'y3', 'y4', 'y5'],
        colormap='Electric'
    )
)

# histogram
vis.histogram(X=np.random.rand(10000), opts=dict(numbins=20))

# bar plots
vis.bar(X=np.random.rand(20))
vis.bar(X=np.abs(np.random.rand(5, 3)),
    opts=dict(
        stacked=True,
        legend=['Facebook', 'Google', 'Twitter'],
        rownames=['2012', '2013', '2014', '2015', '2016']
    )
)
vis.bar(X=np.random.rand(20, 3),
    opts=dict(stacked=False,legend=['The Netherlands', 'France', 'United States'])
)

# scatter plots
Y = np.random.rand(100)
vis.scatter(X=np.random.rand(100, 2),Y=(Y[Y > 0] + 1.5).astype(int),
    opts=dict(
        legend=['Apples', 'Pears'],
        xtickmin=-5,
        xtickmax=5,
        xtickstep=0.5,
        ytickmin=-5,
        ytickmax=5,
        ytickstep=0.5,
        markersymbol='cross-thin-open'
    )
)
vis.scatter(X=np.random.rand(100, 3),Y=(Y + 1.5).astype(int),
    opts=dict(legend=['Men', 'Women'],markersize=5)
)

# image demo
# vis.image(
#    np.random.rand(3, 512, 256),
#    opts=dict(title='Random!', caption='How random.'),
# )

# grid of images
vis.images(np.random.randn(20, 3, 64, 64),
    opts=dict(title='Random images', caption='How random.')
)

print('Done')