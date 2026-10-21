#!/bin/bash
# Fetch the NLCD 2021 land-cover subset covering the Harvey domain from the
# MRLC WCS service, in NLCD's native Albers projection (EPSG:5070).
#
# The bbox covers the Turning_30m mesh (EPSG:32610) with a 2 km pad, converted
# to EPSG:5070. WCS 2.0.1 rejects this server's projection handling, so use
# WCS 1.0.0 and state the CRS and pixel counts explicitly (30 m pixels).
BBOX="-2786,721154,76932,775420"   # xmin,ymin,xmax,ymax in EPSG:5070
curl -s -o nlcd_2021_houston_5070.tif \
  "https://www.mrlc.gov/geoserver/mrlc_display/wcs?service=WCS&version=1.0.0&request=GetCoverage&coverage=NLCD_2021_Land_Cover_L48&bbox=${BBOX}&crs=EPSG:5070&format=GeoTIFF&width=2657&height=1809"
echo "wrote nlcd_2021_houston_5070.tif (pass BBOX to make_nlcd_manning.py)"
