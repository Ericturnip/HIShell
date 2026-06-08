import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

def _ctype_list(hdr):
    n = hdr.get("NAXIS", 0)
    return [str(hdr.get(f"CTYPE{i}", "")).upper() for i in range(1, n+1)]

def _is_spec_ctype(ct: str) -> bool:
    ct = ct.upper()
    # FITS cubes use several spectral-axis names across AIPS and CASA products.
    keys = ("VELO", "VRAD", "VOPT", "FELO", "FREQ", "WAVE", "AWAV", "ZOPT")
    return any(k in ct for k in keys)

def _find_axis_numbers(hdr):
    """
    Return 1-based FITS axis numbers for RA, DEC, and SPECTRAL.
    """
    ctypes = _ctype_list(hdr)
    ra = dec = spec = None
    for i, ct in enumerate(ctypes, start=1):
        if ra is None and ("RA" in ct or "GLON" in ct):
            ra = i
        if dec is None and ("DEC" in ct or "GLAT" in ct):
            dec = i
        if spec is None and _is_spec_ctype(ct):
            spec = i
    return {"ra": ra, "dec": dec, "spec": spec, "naxis": len(ctypes)}

def _axisnum_to_numpy_index(axisnum, naxis):
    return None if axisnum is None else (naxis - axisnum)

def open_cube(path):
    """
    Open a FITS cube and return data in velocity, y, x order.
    Extra axes such as STOKES are dropped by taking index 0.
    """
    hdul = fits.open(path)
    hdr = hdul[0].header
    data = hdul[0].data
    if data is None:
        for h in hdul[1:]:
            if h.data is not None:
                data = h.data
                hdr = h.header
                break
    if data is None:
        raise ValueError(f"No data found in FITS: {path}")

    wcs = WCS(hdr)
    info = _find_axis_numbers(hdr)
    nax = info["naxis"]
    arr = np.asarray(data)

    spec_idx = _axisnum_to_numpy_index(info["spec"], nax)
    ra_idx   = _axisnum_to_numpy_index(info["ra"],   nax)
    dec_idx  = _axisnum_to_numpy_index(info["dec"],  nax)

    # Some headers do not label axes cleanly, so fall back to the last three axes.
    if spec_idx is None or ra_idx is None or dec_idx is None or arr.ndim < 3:
        arr = np.squeeze(arr)
        if arr.ndim < 3:
            raise ValueError(f"Cube must be at least 3D after squeeze; got {arr.shape}")
        v, y, x = arr.shape[-3], arr.shape[-2], arr.shape[-1]
        arr = arr.reshape((-1, v, y, x))[-1]
        return arr.astype(np.float32), hdr, wcs, hdul

    order = [spec_idx, dec_idx, ra_idx] + [i for i in range(arr.ndim) if i not in (spec_idx, dec_idx, ra_idx)]
    arr = np.transpose(arr, order)

    while arr.ndim > 3:
        arr = arr.take(indices=0, axis=3)

    return arr.astype(np.float32), hdr, wcs, hdul

def pixel_scales_arcsec(wcs: WCS):
    """Return (arcsec_per_x, arcsec_per_y) from celestial WCS."""
    cel = wcs.celestial
    scales = proj_plane_pixel_scales(cel) # degrees per pixel, x then y
    return float(abs(scales[0])*3600.0), float(abs(scales[1])*3600.0)

def synthesized_beam_arcsec(hdr):
    """
    Return synthesized beam major/minor FWHM in arcsec from FITS BMAJ/BMIN.

    FITS stores BMAJ/BMIN in degrees. Some local configs carry a fallback
    beam_fwhm_arcsec value, but label auditing should prefer the cube header
    because it is the physical provenance of the PV cut.
    """
    bmaj = hdr.get("BMAJ")
    bmin = hdr.get("BMIN")
    if bmaj is None and bmin is None:
        return None
    if bmaj is None:
        bmaj = bmin
    if bmin is None:
        bmin = bmaj
    major = abs(float(bmaj)) * 3600.0
    minor = abs(float(bmin)) * 3600.0
    if not (np.isfinite(major) and np.isfinite(minor) and max(major, minor) > 0):
        return None
    return max(major, minor), min(major, minor)

def radec_to_xy(wcs: WCS, ra_deg: float, dec_deg: float):
    """RA/Dec (deg) -> pixel (x,y) using celestial WCS."""
    cel = wcs.celestial
    x, y = cel.world_to_pixel_values(ra_deg, dec_deg)
    return float(x), float(y)

def velocity_axis_kms(hdr):
    """
    Build spectral axis in km/s regardless of which CTYPE# it lives on.
    Frequency HI axes are converted with the radio definition
    v = c * (rest - freq) / rest. This matches THINGS HI cubes whose
    CTYPE is FREQ but whose catalog velocities are heliocentric km/s.
    """
    info = _find_axis_numbers(hdr)
    spec_ax = info["spec"]
    if spec_ax is None:
        # Final fallback: if there are 3+ axes, try the last non-spatial axis
        ctypes = _ctype_list(hdr)
        for i, ct in enumerate(ctypes, start=1):
            if not ("RA" in ct or "DEC" in ct or "GLON" in ct or "GLAT" in ct):
                if _is_spec_ctype(ct):
                    spec_ax = i
                    break
        if spec_ax is None:
            raise ValueError("Could not find spectral axis in CTYPE keywords.")

    nv = int(hdr.get(f"NAXIS{spec_ax}"))
    crval = float(hdr.get(f"CRVAL{spec_ax}"))
    cdelt = float(hdr.get(f"CDELT{spec_ax}"))
    crpix = float(hdr.get(f"CRPIX{spec_ax}"))
    cunit = str(hdr.get(f"CUNIT{spec_ax}", "")).lower()
    ctype = str(hdr.get(f"CTYPE{spec_ax}", "")).upper()

    idx = np.arange(nv, dtype=np.float64)
    world = crval + (idx + 1 - crpix) * cdelt  # FITS 1-indexed CRPIX

    if "FREQ" in ctype or "hz" in cunit or "ghz" in cunit or "mhz" in cunit:
        rest = hdr.get("RESTFRQ", hdr.get("RESTFREQ"))
        if rest is None:
            raise ValueError("SPECTRAL axis is frequency but RESTFRQ/RESTFREQ is missing.")
        freq_hz = world.copy()
        if "ghz" in cunit:
            freq_hz *= 1.0e9
        elif "mhz" in cunit:
            freq_hz *= 1.0e6
        # Blank CUNIT in THINGS frequency cubes is already Hz.
        c_kms = 299792.458
        world = c_kms * (float(rest) - freq_hz) / float(rest)
    elif "km/s" in cunit or "kms-1" in cunit or "km s-1" in cunit:
        pass
    elif "m/s" in cunit or "ms-1" in cunit or "m s-1" in cunit:
        world /= 1000.0
    else:
        # Blank or unusual units are common enough to warn without stopping.
        print(f"[velocity_axis_kms] Warning: unknown CUNIT{spec_ax}='{cunit}', assuming km/s.")

    return world.astype(np.float32)

def line_extent_to_bounds(cx, cy, dx, dy, nx, ny):
    """
    Find the parameter range where a line stays inside the image bounds.
    """
    ts = []
    if dx != 0:
        ts += [(0 - cx)/dx, ((nx-1) - cx)/dx]
    if dy != 0:
        ts += [(0 - cy)/dy, ((ny-1) - cy)/dy]
    t_candidates = []
    for t in ts:
        x = cx + t*dx; y = cy + t*dy
        if -1 <= x <= nx and -1 <= y <= ny:
            t_candidates.append(t)
    if not t_candidates:
        return -0.0, 0.0
    tmin, tmax = min(t_candidates), max(t_candidates)
    return tmin, tmax

def rotate_xy(x, y, pa_deg: float, convention: str = "astro"):
    """
    Rotate points by +PA according to the convention above.
    Returns (xr, yr) in the same pixel frame.
    """
    th = np.deg2rad(float(pa_deg))
    c, s = np.cos(th), np.sin(th)
    if convention == "astro":
        # +PA rotates +y toward +x
        # Rotate standard (image) axes so that a positive PA brings +y toward +x
        # Equivalent to using the (sin, cos) mapping used in unit_vectors_for_pa
        xr =  x *  c + y * s
        yr = -x *  s + y * c
    else:  # "image"
        xr =  x *  c - y * s
        yr =  x *  s + y * c
    return xr, yr

def unit_vectors_for_pa(pa_deg: float, convention: str = "astro") -> tuple[tuple[float,float], tuple[float,float]]:
    """
    Returns (dvec, nvec):
      dvec: unit vector along the PA direction on the sky (projected in image pix coords)
      nvec: unit vector perpendicular to dvec (rotate CCW by +90°)
    convention:
      "astro" -> PA measured east of north (CCW from +y axis)
      "image" -> PA measured CCW from +x axis (typical math/image)
    """
    th = np.deg2rad(float(pa_deg))
    if convention == "astro":
        # 0° = +y, 90° = +x
        # A rotation by +th from +y toward +x
        dx =  np.sin(th)
        dy =  np.cos(th)
    elif convention == "image":
        # 0° = +x, 90° = +y
        dx =  np.cos(th)
        dy =  np.sin(th)
    else:
        raise ValueError(f"unknown convention: {convention}")
    # perpendicular, CCW +90°
    nx = -dy
    ny =  dx
    return (dx, dy), (nx, ny)
