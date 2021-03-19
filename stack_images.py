import numpy as np
import sys
import os
import argparse
import glob
import subprocess
import tempfile
import pandas
import re
import xlrd
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy import log
from racs_tools import beamcon_2D
import warnings

table_names = {
    "tiles": "Tile Corrections",
    "combined": "Combined Corrections",
}
column_names = {
    "ra": "ra_correction",
    "dec": "dec_correction",
    "flux_offset": "flux_peak_correction_additive",
    "flux_scale": "flux_peak_correction_multiplicative",
}

warnings.filterwarnings("ignore", category=FITSFixedWarning)


def shift_and_scale_image(
    imagename,
    rmsimagename,
    outputdir,
    flux_scale=1,
    flux_offset=0,
    ra_offset=0,
    dec_offset=0,
    subimage=None,
    squeeze_output=False,
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
    squeeze_output=False,
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
    fimg_ndim = fimg[0].data.ndim
    fimg[0].data = flux_scale * (fimg[0].data + flux_offset)
    fimg[0].header["FLUXOFF"] = flux_offset
    fimg[0].header["FLUXSCL"] = flux_scale
    frms_ndim = frms[0].data.ndim
    frms[0].data = flux_scale * (frms[0].data)
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
    w.wcs.crval[0:2] = np.array([crval_offset.ra.deg, crval_offset.dec.deg])
    newheader = w.to_header()
    # update the header with the new WCS
    fimg[0].header.update(newheader)
    frms[0].header.update(newheader)

    fimg[0].header["RAOFF"] = ra_offset
    fimg[0].header["DECOFF"] = dec_offset
    frms[0].header["RAOFF"] = ra_offset
    frms[0].header["DECOFF"] = dec_offset

    if subimage is not None and subimage >= 0:
        center = [int(x / 2.0) for x in fimg[0].data.shape[-2:]]  # last 2 axis
        log.debug(
            "Extracting subimage of size [%d,%d] around [%d,%d]"
            % (subimage, subimage, center[0], center[1])
        )
        cutout = Cutout2D(fimg[0].data.squeeze(), center, subimage, wcs=w.celestial)
        cutout_rms = Cutout2D(frms[0].data.squeeze(), center, subimage, wcs=w.celestial)
        # replace original data, adding singleton Stokes, Freq dims if necessary
        if not squeeze_output:
            log.debug(
                "Ensuring cutout ndim (%d) matches input image ndim (%d)"
                % (cutout.data.ndim, fimg_ndim)
            )
            for _ in range(fimg_ndim - cutout.data.ndim):
                cutout.data = np.expand_dims(cutout.data, axis=0)
            for _ in range(frms_ndim - cutout_rms.data.ndim):
                cutout_rms.data = np.expand_dims(cutout_rms.data, axis=0)
            log.debug(
                "New cutout ndim: %d cutout rms ndim: %d."
                % (cutout.data.ndim, cutout_rms.data.ndim)
            )
        fimg[0].data = cutout.data
        frms[0].data = cutout_rms.data

        # update the header with the new WCS
        newheader = cutout.wcs.to_header()
        fimg[0].header.update(newheader)
        frms[0].header.update(newheader)
        fimg[0].header.remove("WCSAXES", ignore_missing=True, remove_all=True)
        frms[0].header.remove("WCSAXES", ignore_missing=True, remove_all=True)

    if os.path.exists(outname):
        os.remove(outname)
    if os.path.exists(outweight):
        os.remove(outweight)
    fimg.writeto(outname)
    frms.writeto(outweight)
    del fimg
    del frms
    return outname, outweight


def swarp_files(files, output_file, output_weight, headerinfo={}):
    """
    result = swarp_files(files, output_file, output_weight, headerinfo={})
    returns True if successful
    """
    cmd = "swarp -VMEM_MAX 4095 -MEM_MAX 2048 -COMBINE_BUFSIZE 2048"
    cmd += " -IMAGEOUT_NAME {} -WEIGHTOUT_NAME {}".format(output_file, output_weight)
    cmd += " -COMBINE Y -COMBINE_TYPE WEIGHTED -SUBTRACT_BACK N -WRITE_XML N"
    cmd += " -FSCALASTRO_TYPE NONE"
    cmd += " -WEIGHT_TYPE MAP_RMS -WEIGHT_SUFFIX .weight.fits -RESCALE_WEIGHTS Y"
    cmd += " -PROJECTION_TYPE SIN"
    # cmd += " -CENTER_TYPE MANUAL -CENTER %s -IMAGE_SIZE %d,%d" % (dir_str, nx, ny)
    # cmd += " -PIXELSCALE_TYPE MANUAL -PIXEL_SCALE %.1f" % ps
    cmd += " %s" % (",".join(files))

    log.info("Running:\n\t%s" % cmd)

    p = subprocess.Popen(cmd.split(), stderr=subprocess.PIPE)
    imagenum = 0

    logfile = tempfile.TemporaryFile(mode="w", prefix="swarp")
    for line in p.stderr:
        line_str = line.decode("utf-8")
        if "line:" not in line_str:
            logfile.write(line_str)
        if "-------------- File" in line_str:
            _ = line_str.split()[-1].replace(":", "")
            imagenum += 1
            log.info("Working on image %04d/%04d..." % (imagenum, len(files)))
        if "Co-adding frames" in line_str:
            log.info("Coadding...")
    logfile.close()
    # update header
    if os.path.exists(output_file):
        f = fits.open(output_file, mode="update")
        f[0].header["BUNIT"] = "Jy/beam"
        for k in headerinfo:
            f[0].header[k] = headerinfo[k]
        f.flush()
    return os.path.exists(output_file) and os.path.exists(output_weight)


def measure_beamsize(
    image, weight=None, default_beam=5 * u.arcsec, weightthreshratio=10, clean=True
):
    """
    BMAJ, BMIN, BPA = measure_beamsize(image, weight = None, default_beam = 15*u.arcsec, weightthreshratio = 10, clean=True)

    use Aegean to find and measure sources
    return the median major axis, minor axis, and position angle

    will create BANE background and rms images.   If clean, will delete them after.
    looks at weight image (if supplied).
    Sets a threshold of median(weight)/weightthreshratio for good sources
    """

    import AegeanTools
    from AegeanTools.source_finder import SourceFinder
    from AegeanTools.wcs_helpers import Beam
    from AegeanTools.BANE import filter_image

    basename = os.path.splitext(image)[0]

    # figure out the number of pixels per synthesized beam
    # of course we don't know the synthesized beam, but guess
    f = fits.open(image)
    w = WCS(f[0].header)
    pixels_per_beam = (default_beam / (w.wcs.cd[1, 1] * u.deg)).decompose().value
    # construct a Beam object
    default_beam = Beam(default_beam.to(u.deg).value, default_beam.to(u.deg).value, 0)
    # figure out step/box sizes for BANE
    # defaults are grid = 4*beam size
    # and box = 5*grid
    step_size = (int(pixels_per_beam * 4), int(pixels_per_beam * 4))
    box_size = (step_size[0] * 5, step_size[1] * 5)
    # BANE output
    bkg_image = "{}_bkg.fits".format(basename)
    rms_image = "{}_rms.fits".format(basename)
    if os.path.exists(bkg_image):
        os.remove(bkg_image)
    if os.path.exists(rms_image):
        os.remove(rms_image)
    filter_image(
        im_name=image, step_size=step_size, box_size=box_size, out_base=basename,
    )
    if not os.path.exists(rms_image):
        log.error("Cannot find BANE output '{}'".format(rms_image))
        return None, None, None
    if not os.path.exists(bkg_image):
        log.error("Cannot find BANE output '{}'".format(bkg_image))
        return None, None, None
    log.debug("BANE wrote {}, {}".format(rms_image, bkg_image))
    sf = SourceFinder()
    found = sf.find_sources_in_image(
        image, beam=default_beam, bkgin=bkg_image, rmsin=rms_image
    )
    if len(found) == 0:
        log.error("Aegean did not find any sources")
        return None, None, None
    log.info("Aegean found {} sources".format(len(found)))
    ra = []
    dec = []
    a = []
    b = []
    pa = []
    for s in found:
        a.append(s.a)
        b.append(s.b)
        pa.append(s.pa)
        ra.append(s.ra)
        dec.append(s.dec)
    a = np.array(a) * u.arcsec
    b = np.array(b) * u.arcsec
    pa = np.array(pa) * u.deg
    ra = np.array(ra)
    dec = np.array(dec)
    if weight is not None:
        # we need to look at the weight image to eliminate bad sources around the edges
        fw = fits.open(weight)
        dw = fw[0].data
        dw = dw[dw > 0]
        medweight = np.median(dw)
        # threshold
        weightthresh = medweight / weightthreshratio
        x, y = w.all_world2pix(ra, dec, 0)
        weightval = fw[0].data[np.int16(y), np.int16(x)]
        good = weightval > weightthresh
        log.info(
            "Found {} sources above weight threshold of {:.2e}".format(
                good.sum(), weightthresh
            )
        )
        a = a[good]
        b = b[good]
        pa = pa[good]

    if clean:
        os.remove(bkg_image)
        os.remove(rms_image)
    return np.median(a).to(u.deg), np.median(b).to(u.deg), np.median(pa)


def main():
    log.setLevel("WARNING")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "fields",
        nargs="+",
        type=str,
        help="Field name(s) without survey prefix. e.g. 0012+00A.",
    )
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
        "--offset",
        default=None,
        type=float,
        help="Flux offset [mJy] (default is to use the QC file)",
    )
    parser.add_argument(
        "--scale",
        default=None,
        type=float,
        help="Flux scaling (default is to use the QC file)",
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
        default=False,
        action="store_true",
        help="Do not use flux offset?",
    )
    parser.add_argument(
        "--type",
        dest="imagetype",
        default="tiles",
        choices=["tiles", "combined"],
        help="Type of images to combine",
    )
    parser.add_argument(
        "--squeeze",
        default=False,
        action="store_true",
        help=(
            "Squeeze output image dimensions to celestial axes only, removing singleton "
            "axes (e.g. frequency, Stokes). Default is to match the input image."
        ),
    )
    parser.add_argument(
        "--nosmooth",
        dest="smooth",
        action="store_false",
        default=True,
        help="Do not smooth the input images to a common resolution",
    )
    parser.add_argument(
        "--convmode",
        default="robust",
        choices=["robust", "scipy", "astropy", "astropy_fft"],
        help="Convolution mode",
    )

    parser.add_argument(
        "-b",
        "--beam",
        action="store_true",
        default=False,
        help="Compute median synthesized beam with Aegean?",
    )
    parser.add_argument(
        "-v", "--verbosity", action="count", help="Increase output verbosity"
    )

    args = parser.parse_args()
    if args.verbosity == 1:
        log.setLevel("INFO")
    elif args.verbosity >= 2:
        log.setLevel("DEBUG")

    log.debug("Running\n\t%s" % " ".join(map(str, sys.argv)))

    if not os.path.exists(args.qc):
        raise FileNotFoundError("Cannot open VAST QC file '%s'" % args.qc)

    try:
        table_corrections = Table.from_pandas(
            pandas.read_excel(
                args.qc, sheet_name=table_names[args.imagetype], engine="openpyxl"
            )
        )
        # remove survey prefix from corrections table field names
        table_corrections["field"] = np.array(
            np.char.split(table_corrections["field"].astype(np.str_), "_").tolist()
        )[:, 1]
    except xlrd.biffh.XLRDError:
        log.warning(
            "Unable to read table '%s' from sheet '%s'"
            % (table_names[args.imagetype], args.qc)
        )
        table_corrections = None

    if not (os.path.exists(args.out) and os.path.isdir(args.out)):
        log.info("Creating output directory '%s'" % args.out)
        os.mkdir(args.out)

    todelete = []
    for field in args.fields:
        if args.imagetype == "combined":
            files = sorted(
                glob.glob(os.path.join(args.imagepath, "VAST_{}*I.fits".format(field),))
            ) + sorted(
                glob.glob(os.path.join(args.imagepath, "RACS_{}*I.fits".format(field),))
            )
            rmsmaps = [f.replace("I.fits", "I_rms.fits") for f in files]
            if "STOKESI_IMAGES" in args.imagepath:
                rmsmaps = [
                    f.replace("STOKESI_IMAGES", "STOKESI_RMSMAPS") for f in rmsmaps
                ]
        elif args.imagetype == "tiles":
            # this is VAST and not RACS
            # because we don't have RACS RMS maps
            files = sorted(
                glob.glob(
                    os.path.join(
                        args.imagepath, "*VAST_{}*restored.fits".format(field),
                    )
                )
            )
            rmsmaps = [f.replace("image.i", "noiseMap.image.i") for f in files]
            if "STOKESI_IMAGES" in args.imagepath:
                rmsmaps = [
                    f.replace("STOKESI_IMAGES", "STOKESI_RMSMAPS") for f in rmsmaps
                ]

        log.info("Found %d images for field %s" % (len(files), field))
        log.debug("Images: %s" % ",".join(files))

        # go through and make temporary files with the scales and offsets applied
        # also make the weight maps ~ rms**2
        scaledfiles = []
        weightfiles = []
        headerinfo = {}
        for i, (filename, rmsmap) in enumerate(zip(files, rmsmaps)):
            headerinfo["IMG%02d" % i] = (filename, "Filename for image %02d" % i)
            headerinfo["RMS%02d" % i] = (rmsmap, "RMS Map name for image %02d" % i)
            # might need to make this more robust
            # this is the name of the epoch in the QC table
            # search for the EPOCH string in the entire file path
            # Combined images contain the epoch in the path and filename, tile images
            # contain the epoch in the path.
            match = re.search(r"EPOCH(\d{2})(x?)", filename)
            if match is None:
                log.error("Cannot infer epoch number from file '%s'" % filename)
                continue
            epoch = match.group()
            log.debug(
                "Infered '%s' for file '%s': will look up QC for '%s'"
                % (match.group(), filename, epoch)
            )
            if table_corrections is not None:
                table_corrections_mask = (table_corrections["field"] == field) & (
                    table_corrections["release_epoch"] == epoch
                )

                # positions
                # corrected <ra|dec> = <ra|dec> + <ra|dec>_correction
                ra_offset = table_corrections[table_corrections_mask][
                    column_names["ra"]
                ]
                dec_offset = table_corrections[table_corrections_mask][
                    column_names["dec"]
                ]

                # fluxes
                # corrected flux = flux_scale * (raw flux + flux_offset)
                flux_scale = table_corrections[table_corrections_mask][
                    column_names["flux_scale"]
                ]
                # original offsets are in mJy
                flux_offset = (
                    table_corrections[table_corrections_mask][
                        column_names["flux_offset"]
                    ]
                    * 1e-3
                )
            else:
                ra_offset = np.zeros(1)
                dec_offset = np.zeros(1)
                flux_offset = np.zeros(1)
                flux_scale = np.ones(1)

            if args.nooffset:
                log.debug("Setting offset to 0...")
                flux_offset = np.zeros(len(flux_offset))
            if args.offset is not None:
                log.debug("Setting offset to %.3f mJy..." % args.offset)
                flux_offset = np.ones(len(flux_offset)) * args.offset
            if args.scale is not None:
                log.debug("Setting scale to %.3f..." % args.scale)
                flux_scale = np.ones(len(flux_offset)) * args.scale

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
                    (
                        "Could not find entries for Field %s and epoch %s; "
                        "assuming unity scale/0 offset"
                    )
                    % (field, epoch)
                )
                flux_scale = 1
                flux_offset = 0
                ra_offset = 0
                dec_offset = 0
            headerinfo["RAOFF%02d" % i] = (
                ra_offset,
                "[ARCSEC] RA Offset for image %02d" % i,
            )
            headerinfo["DECOFF%02d" % i] = (
                dec_offset,
                "[ARCSEC] DEC Offset for image %02d" % i,
            )
            headerinfo["FLXSCL%02d" % i] = (flux_scale, "Flux scale for image %02d" % i)
            headerinfo["FLXOFF%02d" % i] = (
                flux_offset,
                "[Jy] Flux offset for image %02d" % i,
            )

            outname, outweight = shift_and_scale_image(
                filename,
                rmsmap,
                args.out,
                flux_scale=flux_scale,
                flux_offset=flux_offset,
                ra_offset=ra_offset,
                dec_offset=dec_offset,
                subimage=args.subimage,
                squeeze_output=args.squeeze,
            )

            log.info("Wrote %s and %s" % (outname, outweight))
            todelete.append(outname)
            todelete.append(outweight)
            scaledfiles.append(outname)
            weightfiles.append(outweight)

        output_file = os.path.join(args.out, "{}_mosaic.fits".format(field))
        output_weight = output_file.replace("_mosaic.fits", "_weight.fits")

        if args.smooth:
            # convolve up to a single beam size
            # first, get the beam
            convolution_mode = "robust"
            beamsuffix = "smooth"
            # Find smallest common beam
            big_beam, allbeams = beamcon_2D.getmaxbeam(files, conv_mode=args.convmode,)
            log.info("Common beam size is: {}".format(str(big_beam)))

            scaled_smoothed_files = []
            weight_smoothed_files = []
            for filename, weightname in zip(scaledfiles, weightfiles):
                # figure out the output filenames
                output_scaledfile = filename.replace(
                    ".fits", ".{}.fits".format(beamsuffix)
                )
                output_scaledweight = weightname.replace(
                    ".weight.fits", ".{}.weight.fits".format(beamsuffix)
                )
                # get some metadata
                datadict = beamcon_2D.getimdata(filename)
                # figure out the convolving beam needed to achieve
                # the desired final beam
                # along with the scaling factor
                conbeam, sfactor = beamcon_2D.getbeam(datadict, big_beam)
                datadict.update(
                    {"conbeam": conbeam, "final_beam": big_beam, "sfactor": sfactor}
                )
                # actually do the smoothing
                newim = beamcon_2D.smooth(datadict, conv_mode=args.convmode)
                # output results
                f = fits.open(filename)
                f[0].header["ORIGBMAJ"] = (f[0].header["BMAJ"], "[deg] Original BMAJ")
                f[0].header["ORIGBMIN"] = (f[0].header["BMIN"], "[deg] Original BMIN")
                f[0].header["ORIGBPA"] = (f[0].header["BPA"], "[deg] Original BPA")
                f[0].header = datadict["final_beam"].attach_to_header(f[0].header)
                f[0].header["CONVBMAJ"] = (
                    conbeam.major.to(u.deg).value,
                    "[deg] Convolving BMAJ",
                )
                f[0].header["CONVBMIN"] = (
                    conbeam.minor.to(u.deg).value,
                    "[deg] Convolving BMIN",
                )
                f[0].header["CONVBPA"] = (
                    conbeam.pa.to(u.deg).value,
                    "[deg] Convolving BPA",
                )
                f[0].data = newim
                f[0].header["BMSCALE"] = (sfactor, "Beam area scaling factor")
                f.writeto(output_scaledfile, overwrite=True)
                log.info("Wrote convolved image to {}".format(output_scaledfile))
                datadict = beamcon_2D.getimdata(weightname)
                conbeam, sfactor = beamcon_2D.getbeam(datadict, big_beam)
                datadict.update(
                    {"conbeam": conbeam, "final_beam": big_beam, "sfactor": sfactor}
                )
                newim = beamcon_2D.smooth(datadict, conv_mode=args.convmode)
                f = fits.open(weightname)
                f[0].header["ORIGBMAJ"] = f[0].header["BMAJ"]
                f[0].header["ORIGBMIN"] = f[0].header["BMIN"]
                f[0].header["ORIGBPA"] = f[0].header["BPA"]
                f[0].header = datadict["final_beam"].attach_to_header(f[0].header)
                f[0].header["CONVBMAJ"] = conbeam.major.to(u.deg).value
                f[0].header["CONVBMIN"] = conbeam.minor.to(u.deg).value
                f[0].header["CONVBPA"] = conbeam.pa.to(u.deg).value
                f[0].data = newim
                f[0].header["BMSCALE"] = (sfactor, "Beam area scaling factor")
                f.writeto(output_scaledweight, overwrite=True)
                log.info(
                    "Wrote convolved weight image to {}".format(output_scaledweight)
                )

                scaled_smoothed_files.append(output_scaledfile)
                weight_smoothed_files.append(output_scaledweight)
                todelete.append(output_scaledfile)
                todelete.append(output_scaledweight)

            scaledfiles = scaled_smoothed_files
            weightfiles = weight_smoothed_files

        if (args.suffix is not None) and (len(args.suffix) > 0):
            output_file = output_file.replace(".fits", "_{}.fits".format(args.suffix))
            output_weight = output_weight.replace(
                ".fits", "_{}.fits".format(args.suffix)
            )

        if args.progressive:
            istart = 0
            for iend in range(istart + 1, len(files)):
                log.info("Swarping files %02d to %02d" % (istart, iend))
                output_file_step = output_file.replace(
                    ".fits", "_%02d-%02d.fits" % (istart, iend)
                )
                output_weight_step = output_file_step.replace("_mosaic", "_weight")
                result = swarp_files(
                    scaledfiles[istart:iend],
                    output_file_step,
                    output_weight_step,
                    headerinfo,
                )
                if result:
                    log.info("Wrote %s and %s" % (output_file_step, output_weight_step))

        # to get swarp:
        # module use /sharedapps/LS/cgca/modulefiles/
        # module load swarp
        result = swarp_files(
            scaledfiles, output_file, output_weight, headerinfo=headerinfo
        )
        if result:
            log.info("Wrote %s and %s" % (output_file, output_weight))
        else:
            log.warning("Error writing %s and %s" % (output_file, output_weight))

        if args.beam:
            BMAJ, BMIN, BPA = measure_beamsize(
                output_file, output_weight, clean=args.clean
            )
            if BMAJ is not None:
                f = fits.open(output_file, mode="update")
                f[0].header["BMAJ"] = (BMAJ.to(u.deg).value, "Median BMAJ")
                f[0].header["BMIN"] = (BMIN.to(u.deg).value, "Median BMIN")
                f[0].header["BPA"] = (BPA.to(u.deg).value, "Median BPA")
                f.flush()
                log.info(
                    "Updated BMAJ = {:.4f} deg, BMIN = {:.4f} deg, BPA = {:.1f} deg in {}".format(
                        BMAJ.to(u.deg).value,
                        BMIN.to(u.deg).value,
                        BPA.to(u.deg).value,
                        output_file,
                    )
                )

        if args.clean:
            for filename in todelete:
                log.debug("Removing temporary file '%s'" % filename)
                os.remove(filename)


if __name__ == "__main__":
    main()
