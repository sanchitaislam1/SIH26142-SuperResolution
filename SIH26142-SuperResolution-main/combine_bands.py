import rasterio

B04 = "data/B04.tif"
B03 = "data/B03.tif"
B02 = "data/B02.tif"
B08 = "data/B08.tif"

OUTPUT = "data/sentinel2_4band.tif"

with rasterio.open(B04) as red, \
     rasterio.open(B03) as green, \
     rasterio.open(B02) as blue, \
     rasterio.open(B08) as nir:

    profile = red.profile.copy()
    profile.update(
        count=4,
        dtype=red.dtypes[0]
    )

    with rasterio.open(OUTPUT, "w", **profile) as dst:
        dst.write(red.read(1), 1)    # B04 Red
        dst.write(green.read(1), 2)  # B03 Green
        dst.write(blue.read(1), 3)   # B02 Blue
        dst.write(nir.read(1), 4)    # B08 NIR

print("Created:", OUTPUT)