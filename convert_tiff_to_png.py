from PIL import Image

# Open the TIFF image
tiff_image = Image.open("/localscratch/pokemon/conversion/HM25_HighRes_Aligned0050(1).tif")

# Convert the image to PNG format
jpeg_image = tiff_image.convert("RGB")

# Save the PNG image
jpeg_image.save("HM25_HighRes_Aligned0050(1).png")
print('done')   # Done