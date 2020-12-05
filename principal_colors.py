import sys
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.image as mpimg

class Point:

  def __init__(self, coordinates):
    self.coordinates = coordinates


class Cluster:

    def __init__(self, center, points):
        self.center = center
        self.points = points


def get_points(image_path):
    img = Image.open(image_path)
    img.thumbnail((200, 400))
    img = img.convert('RGB')
    w, h = img.size

    points = []

    for count, color in img.getcolors(w*h):
        for _ in range(count):
            points.append(Point(color))

    return points

def euclidean(p, q):
  n_dim = len(p.coordinates)
  return np.sqrt(sum([
      (p.coordinates[i] - q.coordinates[i]) ** 2 for i in range(n_dim)
  ]))



class KMeans:
    def __init__(self, n_clusters, min_diff):
        self.n_clusters = n_clusters
        self.min_diff = min_diff

    def calculate_center(self, points):
        n_dim = len(points[0].coordinates)
        vals = [0.0 for i in range(n_dim)]
        for p in points:
          for i in range(n_dim):
            vals[i] += p.coordinates[i]
        coords = [(v / len(points)) for v in vals]
        return Point(coords)

    def assign_points(self, clusters, points):
        plists = [[] for i in range(self.n_clusters)]

        for p in points:
            smallest_distance = float('inf')

            for i in range(self.n_clusters):
                distance = euclidean(p, clusters[i].center)
                if distance < smallest_distance:
                    smallest_distance = distance
                    idx = i

            plists[idx].append(p)

        return plists
 
    
    def fit(self, points):
        clusters = [Cluster(center=p, points=[p]) for p in random.sample(points, self.n_clusters)]

        while True:

            plists = self.assign_points(clusters, points)

            diff = 0

            for i in range(self.n_clusters):
                if not plists[i]:
                    continue
                old = clusters[i]
                center = self.calculate_center(plists[i])
                new = Cluster(center, plists[i])
                clusters[i] = new
                diff = max(diff, euclidean(old.center, new.center))

            if diff < self.min_diff:
                break

        return clusters

def plotgrid(listcolors, path_image):
    #fig, ax = plt.subplots()
    #ax.pcolormesh(np.random.rand(20,20), cmap='hot')

    ny, nx = 1, len(listcolors)
    #r, g, b = [np.random.random(ny*nx).reshape((ny, nx)) for _ in range(3)]
    fig, (image_axis, pallete_axis) = plt.subplots(2, 1)
    #c = np.dstack([r,g,b])
    #c = np.vstack(listcolors)
    c = np.dstack(listcolors).transpose(0, 2, 1)
    image   =   mpimg.imread(path_image)
    image_axis.imshow(image)
    image_axis.set_xticks([])
    image_axis.set_yticks([])
    pallete_axis.imshow(c, interpolation='nearest')
    plt.show()

    #ax.imshow(data, cmap=cmap, norm=norm)


def rgb_to_hex(rgb):
  return '#%s' % ''.join(('%02x' % p for p in rgb))

def get_colors(filename, n_colors=3, min_diff=20):
  points = get_points(filename)
  clusters = KMeans(n_clusters=n_colors, min_diff=min_diff).fit(points)
  clusters.sort(key=lambda c: len(c.points), reverse = True)
  rgbs = [np.array(c.center.coordinates).astype(np.int) for c in clusters]
  #rgbs = [map(int, c.center.coordinates) for c in clusters]
  return rgbs#list(map(rgb_to_hex, rgbs))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        imagename = sys.argv[1]
    else:
        imagename = "spline_test/spline1.png"
        print("You don't specified the image to process, by default we choose <spline_test/spline1.png>. You can run your code as follows:\n\n$ python principal_colors.py {}\n".format(imagename))
        print('--'*10)
    colors = get_colors(imagename, n_colors=6, min_diff=15)
    plotgrid(colors, imagename)