from astropy.table import Table
import pandas
import os
import sys
import argparse
import numpy as np
from astropy.io import fits
from astropy import log


def measure_beam(selavy_catalog, fitsfile=None):

    # read in the raw data
    log.info("Reading {}...".format(selavy_catalog))
    data = Table.from_pandas(pandas.read_fwf(selavy_catalog, skiprows=[1,]))
    # median seems to work well
    # make sure to convert from arcsec to degrees
    BMAJ = np.median(data["maj_axis"]) / 3600
    BMIN = np.median(data["min_axis"]) / 3600
    BPA = np.median(data["pos_ang"])

    log.info(
        "Measured BMAJ = {:.1f} arcsec from {} sources".format(BMAJ * 3600, len(data))
    )
    log.info(
        "Measured BMIN = {:.1f} arcsec from {} sources".format(BMIN * 3600, len(data))
    )
    log.info("Measured BPA = {:.1f} deg from {} sources".format(BPA, len(data)))

    if fitsfile is not None:
        f = fits.open(fitsfile, mode="update")
        f[0].header["BMAJ"] = (BMAJ, "Median of {}".format(selavy_catalog))
        f[0].header["BMIN"] = (BMIN, "Median of {}".format(selavy_catalog))
        f[0].header["BPA"] = (BPA, "Median of {}".format(selavy_catalog))
        f.flush()
        log.info(
            "Updated BMAJ = {:.4f} deg, BMIN = {:.4f} deg, BPA = {:.1f} deg in {}".format(
                BMAJ, BMIN, BPA, fitsfile
            )
        )

    return BMAJ, BMIN, BPA


def main():
    log.setLevel("WARNING")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("catalog", type=str, help="Selavy catalog name")
    parser.add_argument(
        "-i", "--image", default=None, type=str, help="FITS image to update"
    )

    parser.add_argument(
        "-v", "--verbosity", action="count", default=0, help="Increase output verbosity"
    )

    args = parser.parse_args()
    if args.verbosity == 1:
        log.setLevel("INFO")
    elif args.verbosity >= 2:
        log.setLevel("DEBUG")

    bmaj, bmin, bpa = measure_beam(args.catalog, fitsfile=args.image)


if __name__ == "__main__":
    main()
