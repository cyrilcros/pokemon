from matplotlib import pyplot
import tifffile

pyplot.imshow(tifffile.imread("train/masks/HM25_HighRes_Aligned_0000_004-_cropped1k0040_0000.png"))
pyplot.show()