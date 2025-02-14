#!/bin/bash

## download the ACDC dataset
ACDC_DOWNLOAD_LINK=https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download
wget $ACDC_DOWNLOAD_LINK -O acdc.zip
mkdir data
unzip acdc.zip -d data/


