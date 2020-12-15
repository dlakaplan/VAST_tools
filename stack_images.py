import numpy as np
import sys
import os
import argparse
import glob
import subprocess
import tempfile
import pandas
from astropy import units as u, constants as c
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy import log


def shift_and_scale_image(
    imagename,
    rmsimagename,
    outputdir,
    flux_scale=1,
    flux_offset=0,
    ra_offset=0,
    dec_offset=0,
    subimage=None,
):
    """
    outimage,outweight = shift_and_scale_image(
    imagename,
    rmsimagename,
    outputdir,
    flux_scale=1,
    flux_offset=0,
    ra_offset=0,
    dec_offset=0,
    subimage=None,
    )

    flux_offset in Jy
    ra_offset/dec_offset in arcsec
    """

    outname = os.path.join(
        outputdir, os.path.split(imagename)[-1].replace("I.fits", "I.image.fits")
    )
    outweight = outname.replace(".fits", ".weight.fits")

    fimg = fits.open(imagename)
    frms = fits.open(rmsimagename)
    # do the flux scaling
    fimg[0].data = (fimg[0].data - flux_offset) / flux_scale
    fimg[0].header["FLUXOFF"] = flux_offset
    fimg[0].header["FLUXSCL"] = flux_scale
    frms[0].data = (frms[0].data) / flux_scale
    frms[0].header["FLUXOFF"] = flux_offset
    frms[0].header["FLUXSCL"] = flux_scale
    w = WCS(fimg[0].header)
    # add the offsets to correct the positions
    # use SkyCoord to handle units and wraps
    # the new coordinates should be old coordintes + offset
    crval = SkyCoord(w.wcs.crval[0] * u.deg, w.wcs.crval[1] * u.deg)
    crval_offset = SkyCoord(
        crval.ra + ra_offset * u.arcsec / np.cos(crval.dec),
        crval.dec + dec_offset * u.arcsec,
    )
    w.wcs.crval = np.array([crval_offset.ra.value, crval_offset.dec.value])
    newheader = w.to_header()
    # update the header with the new WCS
    fimg[0].header.update(newheader)
    frms[0].header.update(newheader)

    fimg[0].header["RAOFF"] = ra_offset
    fimg[0].header["DECOFF"] = dec_offset
    frms[0].header["RAOFF"] = ra_offset
    frms[0].header["DECOFF"] = dec_offset

    if subimage is not None and subimage >= 0:
        center = [int(x / 2.0) for x in fimg[0].data.shape]
        log.debug(
            "Extracting subimage of size [{:d},{:d}] around [{:d},{:d}]".format(
                subimage, subimage, center[0], center[1]
            )
        )
        cutout = Cutout2D(fimg[0].data, center, [subimage, subimage], wcs=w)
        cutout_rms = Cutout2D(frms[0].data, center, [subimage, subimage], wcs=w)
        fimg[0].data = cutout.data
        frms[0].data = cutout_rms.data

        # update the header with the new WCS
        newheader = cutout.wcs.to_header()
        fimg[0].header.update(newheader)
        frms[0].header.update(newheader)

    if os.path.exists(outname):
        os.remove(outname)
    if os.path.exists(outweight):
        os.remove(outweight)
    fimg.writeto(outname)
    frms.writeto(outweight)
    del fimg
    del frms
    return outname, outweight


def swarp_files(files, output_file, output_weight):
    """
    result = swarp_files(files, output_file, output_weight)
    returns True if successful
    """
    cmd = "swarp -VMEM_MAX 4095 -MEM_MAX 2048 -COMBINE_BUFSIZE 2048"
    cmd += " -IMAGEOUT_NAME {} -WEIGHTOUT_NAME {}".format(output_file, output_weight)
    cmd += " -COMBINE Y -COMBINE_TYPE WEIGHTED -SUBTRACT_BACK N -WRITE_XML N"
    cmd += " -FSCALASTRO_TYPE NONE"
    cmd += " -WEIGHT_TYPE MAP_RMS -WEIGHT_SUFFIX .weight.fits -RESCALE_WEIGHTS Y"
    cmd += " -PROJECTION_TYPE SIN "  # -CENTER_TYPE MANUAL -CENTER %s -IMAGE_SIZE %d,%d -PIXELSCALE_TYPE MANUAL -PIXEL_SCALE %.1f " %(dir_str, nx, ny, ps)
    cmd += " %s" % (",".join(files))

    log.info("Running:\n\t%s" % cmd)

    p = subprocess.Popen(cmd.split(), stderr=subprocess.PIPE)
    imagenum = 0

    logfile = tempfile.TemporaryFile(mode="w", prefix="swarp")
    for line in p.stderr:
        line_str = line.decode("utf-8")
        if not "line:" in line_str:
            logfile.write(line_str)
        if "-------------- File" in line_str:
            currentimage = line_str.split()[-1].replace(":", "")
            imagenum += 1
            log.info("Working on image %04d/%04d..." % (imagenum, len(files)))
        if "Co-adding frames" in line_str:
            log.info("Coadding...")
    logfile.close()
    # update header
    if os.path.exists(output_file):
        f=fits.open(output_file,mode='update')
        f[0].header['BUNIT']='Jy/beam'
        f.flush()
    return os.path.exists(output_file) and os.path.exists(output_weight)


def main():
    log.setLevel("WARNING")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("fields", nargs="+", type=str, help="Field name(s)")
    parser.add_argument(
        "-o", "--out", default="temp", type=str, help="Ouptut directory",
    )
    parser.add_argument(
        "-i",
        "--imagepath",
        default=os.path.curdir,
        type=str,
        help="VAST image directories",
    )
    parser.add_argument(
        "-q",
        "--qc",
        default="VAST Pilot QC Stats.xlsx",
        type=str,
        help="VAST Pilot QC file",
    )
    parser.add_argument(
        "-s",
        "--subimage",
        default=None,
        type=int,
        help="Size of subimage (if specified)",
    )
    parser.add_argument(
        "-p",
        "--progressive",
        default=False,
        action="store_true",
        help="Do progressive stack (incrementally output results)?",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        type=str,
        help="Suffix for output images [default=None]",
        )
    parser.add_argument(
        "-c",
        "--clean",
        default=False,
        action="store_true",
        help="Clean temporary files?",
    )
    parser.add_argument(
        "--nooffset",
        default=False
        action="store_true",
        help="Do not use flux offset?",
        )
    parser.add_argument(
        "-v", "--verbosity", action="count", help="Increase output verbosity"
    )

    args = parser.parse_args()
    if args.verbosity == 1:
        log.setLevel("INFO")
    elif args.verbosity >= 2:
        log.setLevel("DEBUG")

    log.debug("Running\n\t%s" % " ".join(map(str,sys.argv)))
        
    if not os.path.exists(args.qc):
        raise FileError("Cannot open VAST QC file '%s'" % args.qc)

    table_offsets = Table.from_pandas(
        pandas.read_excel(args.qc, sheet_name="Positional Offsets Combined")
    )
    table_fluxes = Table.from_pandas(
        pandas.read_excel(args.qc, sheet_name="Flux Ratios Combined")
    )

    if not (os.path.exists(args.out) and os.path.isdir(args.out)):
        log.info("Creating output directory '%s'" % args.out)
        os.mkdir(args.out)

    todelete = []
    for field in args.fields:
        files = sorted(
            glob.glob(os.path.join(args.imagepath, "VAST_{}*I.fits".format(field),))
        )
        rmsmaps = [f.replace("I.fits", "I_rms.fits") for f in files]
        if "STOKESI_IMAGES" in args.imagepath:
            rmsmaps = [f.replace("STOKESI_IMAGES", "STOKESI_RMSMAPS") for f in rmsmaps]

        log.info("Found {} images for field {}".format(len(files), field))
        log.debug("Images: {}".format(",".join(files)))
        
        # go through and make temporary files with the scales and offsets applied
        # also make the weight maps ~ rms**2
        for filename, rmsmap in zip(files, rmsmaps):
            # might need to make this more robust
            # this is the name of the epoch in the QC table
            epoch = (
                os.path.split(filename)[-1]
                .split(".")[1]
                .replace("EPOCH", "vastp")
                .replace("p0", "p")
            )
            # corrected flux = (raw flux - offset) / gradient
            flux_scale = table_fluxes[
                (table_fluxes["image"] == field) & (table_fluxes["epoch"] == epoch)
            ]["flux_ratio_fitted_gradient"]
            # original offsets are in mJy
            flux_offset = (
                table_fluxes[
                    (table_fluxes["image"] == field) & (table_fluxes["epoch"] == epoch)
                ]["flux_ratio_fitted_offset"]
                * 1e-3
            )
            if args.nooffset:
                log.debug("Setting offset to 0...")
                flux_offset = 0
            ra_offset = table_offsets[
                (table_offsets["image"] == field) & (table_offsets["epoch"] == epoch)
            ]["ra_offset_median"]
            dec_offset = table_offsets[
                (table_offsets["image"] == field) & (table_offsets["epoch"] == epoch)
            ]["dec_offset_median"]
            if (
                len(flux_scale) == 1
                and len(flux_offset) == 1
                and len(ra_offset) == 1
                and len(dec_offset) == 1
            ):
                flux_scale = flux_scale[0]
                flux_offset = flux_offset[0]
                ra_offset = ra_offset[0]
                dec_offset = dec_offset[0]

            else:
                log.warning(
                    "Could not find entries for Field {} and epoch {}; assuming unity scale/0 offset".format(
                        field, epoch
                    )
                )
                flux_scale = 1
                flux_offset = 0
                ra_offset = 0
                dec_offset = 0

            outname, outweight = shift_and_scale_image(
                filename,
                rmsmap,
                args.out,
                flux_scale=flux_scale,
                flux_offset=flux_offset,
                ra_offset=ra_offset,
                dec_offset=dec_offset,
                subimage=args.subimage,
            )

            log.info("Wrote {} and {}".format(outname, outweight))
            todelete.append(outname)
            todelete.append(outweight)

        files = glob.glob("{}/*{}*.image.fits".format(args.out,field))
        output_file = os.path.join(args.out, "{}_mosaic.fits".format(field))
        output_weight = output_file.replace("_mosaic.fits", "_weight.fits")

        if (args.suffix is not None) and (len(args.suffix)>0):
            output_file=output_file.replace(".fits","_{}.fits".format(args.suffix))
            output_weight=output_weight.replace(".fits","_{}.fits".format(args.suffix))
        
        if args.progressive:
            istart = 0
            for iend in range(istart + 1, len(files)):
                log.info("Swarping files %02d to %02d" % (istart, iend))
                output_file_step = output_file.replace(
                    ".fits", "_%02d-%02d.fits" % (istart, iend)
                )
                output_weight_step = output_file_step.replace("_mosaic", "_weight")
                result = swarp_files(
                    files[istart:iend], output_file_step, output_weight_step
                )
                if result:
                    log.info(
                        "Wrote {} and {}".format(output_file_step, output_weight_step)
                    )

        # to get swarp:
        # module use /sharedapps/LS/cgca/modulefiles/
        # module load swarp
        result = swarp_files(files, output_file, output_weight)
        if result:
            log.info("Wrote {} and {}".format(output_file, output_weight))
        else:
            log.warning("Error writing {} and {}".format(output_file, output_weight))


        if args.clean:
            for filename in todelete:
                log.debug("Removing temporary file '%s'" % filename)
                os.remove(filename)


if __name__ == "__main__":
    main()
