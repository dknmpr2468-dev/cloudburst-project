"""
static_extract_v2.py
Whole-India terrain layers WITHOUT loading the full DEM into RAM.

Track A (native 30 m, streaming NumPy — constant ~1 GB RAM):
    elevation, slope, aspect, plan_curvature, profile_curvature

Track B (downsampled DEM + WhiteboxTools — flow routing is global, cannot be tiled):
    flow_accumulation, spi, twi, drainage_density

Requires: rasterio, numpy, scipy, whitebox  (no GDAL CLI needed)
"""

import os, sys, shutil
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_origin
# from scipy.ndimage import uniform_filter
# import whitebox

# ====================== CONFIG ======================
dem_path           = "INDIA2000_dem.tif"
output_dir         = "./terrain_outputs/"
suffix             = ""            # e.g. "_cop" for Copernicus runs

tile               = 4096          # processing window (px)
halo               = 2             # overlap px for derivatives

# Hydro DEM resolution (degrees). Flow accumulation MUST fit in RAM (WBT loads all).
#   0.00208333 (~250 m)  -> ~2.4e8 cells  -> WBT needs ~6-10 GB RAM   [safe default]
#   0.00083333 (~90 m)   -> ~1.6e9 cells  -> WBT needs ~40-60 GB RAM  [only big machines]
hydro_res_deg      = 0.00208333

stream_threshold   = 1000          # flow-acc cells defining a stream (at hydro res!)
drainage_window_px = 101           # odd; window for drainage density (at hydro res)

M_PER_DEG = 111320.0
NODATA    = -9999.0                # used for WBT inputs (WBT dislikes NaN nodata)
# ====================================================


def p(name):
    base, ext = os.path.splitext(name)
    return os.path.abspath(os.path.join(output_dir, f"{base}{suffix}{ext}"))


def out_profile(src_profile, nodata=np.nan):
    pr = src_profile.copy()
    pr.update(dtype="float32", count=1, nodata=nodata,
              tiled=True, blockxsize=512, blockysize=512,
              compress="deflate", predictor=3, BIGTIFF="YES")
    return pr


def check(path, step):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] {step} failed — output missing: {path}. "
                 f"Check WhiteboxTools output above (likely RAM — lower hydro_res_deg).")


# ─────────────── Track A: streaming local derivatives ───────────────

def local_derivatives(dem, out_slope, out_aspect=None, out_plan=None,
                      out_prof=None, nodata_out=np.nan):
    """Windowed slope/aspect/curvature for geographic-CRS DEMs of any size."""
    with rasterio.open(dem) as src:
        H, W, T, nd = src.height, src.width, src.transform, src.nodata
        prof = out_profile(src.profile, nodata=nodata_out)
        targets = {"slope": out_slope, "aspect": out_aspect,
                   "plan": out_plan, "prof": out_prof}
        dsts = {k: rasterio.open(f, "w", **prof)
                for k, f in targets.items() if f}

        n_ty = (H + tile - 1) // tile
        n_tx = (W + tile - 1) // tile
        done = 0
        for r0 in range(0, H, tile):
            for c0 in range(0, W, tile):
                rr0, cc0 = max(r0 - halo, 0), max(c0 - halo, 0)
                rr1, cc1 = min(r0 + tile + halo, H), min(c0 + tile + halo, W)
                Z = src.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0)
                             ).astype(np.float64)
                if nd is not None:
                    Z[Z == nd] = np.nan

                h_out = min(tile, H - r0)
                w_out = min(tile, W - c0)
                wout  = Window(c0, r0, w_out, h_out)

                if not np.any(np.isfinite(Z)):
                    blk = np.full((h_out, w_out), nodata_out, np.float32)
                    for d in dsts.values():
                        d.write(blk, 1, window=wout)
                    done += 1
                    continue

                # metre spacing per row (CRS is degrees, dx shrinks with latitude)
                rows = np.arange(rr0, rr1)
                lat  = T.f + T.e * (rows + 0.5)
                dy   = abs(T.e) * M_PER_DEG
                dx   = (abs(T.a) * M_PER_DEG * np.cos(np.radians(lat)))[:, None]

                dZc = np.gradient(Z, axis=1)
                dZr = np.gradient(Z, axis=0)
                p_  =  dZc / dx                       # east
                q_  = -dZr / dy                       # north (rows go south)
                g2  = p_ * p_ + q_ * q_

                res = {}
                res["slope"] = np.degrees(np.arctan(np.sqrt(g2)))
                if "aspect" in dsts:
                    asp = (np.degrees(np.arctan2(-p_, -q_)) + 360.0) % 360.0
                    res["aspect"] = np.where(g2 == 0, -1.0, asp)
                if "plan" in dsts or "prof" in dsts:
                    r_ =  np.gradient(dZc, axis=1) / dx ** 2
                    t_ =  np.gradient(dZr, axis=0) / dy ** 2
                    s_ = -np.gradient(dZc, axis=0) / (dx * dy)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        plan = -(q_*q_*r_ - 2*p_*q_*s_ + p_*p_*t_) / np.power(g2, 1.5)
                        prf  = -(p_*p_*r_ + 2*p_*q_*s_ + q_*q_*t_) / (g2 * np.power(1 + g2, 1.5))
                    res["plan"] = np.where(g2 == 0, 0.0, plan)
                    res["prof"] = np.where(g2 == 0, 0.0, prf)

                ri, ci = r0 - rr0, c0 - cc0
                for k, d in dsts.items():
                    a = res[k][ri:ri + h_out, ci:ci + w_out].astype(np.float32)
                    a[~np.isfinite(Z[ri:ri + h_out, ci:ci + w_out])] = nodata_out
                    if not np.isnan(nodata_out):
                        a = np.nan_to_num(a, nan=nodata_out)
                    d.write(a, 1, window=wout)

                done += 1
                print(f"\r    tiles {done}/{n_ty * n_tx}", end="", flush=True)
        print()
        for d in dsts.values():
            d.close()


# ─────────────── Track B: downsample + WBT hydrology ───────────────

def downsample_dem(dem, dst_path, res):
    """Streaming average-resample via WarpedVRT. Writes NODATA=-9999 for WBT."""
    with rasterio.open(dem) as src:
        b  = src.bounds
        Wn = int(round((b.right - b.left) / res))
        Hn = int(round((b.top - b.bottom) / res))
        tr = from_origin(b.left, b.top, res, res)
        with WarpedVRT(src, transform=tr, width=Wn, height=Hn,
                       resampling=Resampling.average) as vrt:
            prof = out_profile(src.profile, nodata=NODATA)
            prof.update(width=Wn, height=Hn, transform=tr)
            with rasterio.open(dst_path, "w", **prof) as dst:
                for r0 in range(0, Hn, tile):
                    for c0 in range(0, Wn, tile):
                        w = Window(c0, r0, min(tile, Wn - c0), min(tile, Hn - r0))
                        a = vrt.read(1, window=w).astype(np.float32)
                        if src.nodata is not None:
                            a[a == src.nodata] = NODATA
                        a = np.nan_to_num(a, nan=NODATA)
                        dst.write(a, 1, window=w)
        return Hn, Wn


def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.profile


# ─────────────── main ───────────────

def main():
    if not os.path.isfile(dem_path):
        sys.exit(f"[ERROR] DEM not found: {dem_path}")
    os.makedirs(output_dir, exist_ok=True)
    abs_dem = os.path.abspath(dem_path)

    with rasterio.open(abs_dem) as src:
        cells = src.height * src.width
        res_deg = abs(src.transform.a)
        mean_lat = (src.bounds.top + src.bounds.bottom) / 2.0
    print(f"DEM: {src.width} x {src.height} = {cells/1e9:.1f}B cells "
          f"(WBT full-load would need ~{cells*8/1e9:.0f} GB -> streaming instead)")
    print(f"Pixel size: {res_deg:.6f} DEGREES (~{res_deg*M_PER_DEG:.0f} m at equator)\n")

    # 1. Elevation (copy)
    print("[1/9] Elevation …")
    shutil.copy(abs_dem, p("elevation.tif"))

    # 2-5. Slope / aspect / curvatures at NATIVE resolution, streaming
    print("[2-5/9] Slope + aspect + plan/profile curvature (streaming, native res) …")
    local_derivatives(abs_dem,
                      out_slope=p("slope.tif"),
                    #   out_aspect=p("aspect.tif"),
                    #   out_plan=p("plan_curvature.tif"),
                    #   out_prof=p("profile_curvature.tif")
                    )

    # # 6a. Downsampled hydro DEM
    # print(f"[6/9] Downsampling DEM to {hydro_res_deg:.6f}° "
    #       f"(~{hydro_res_deg*M_PER_DEG:.0f} m) for hydrology …")
    # hydro_dem = p("_dem_hydro.tif")
    # Hn, Wn = downsample_dem(abs_dem, hydro_dem, hydro_res_deg)
    # est = Hn * Wn * 8 / 1e9
    # print(f"    hydro DEM: {Wn} x {Hn}  (WBT will need roughly {est*3:.0f}-{est*6:.0f} GB RAM)")

    # # slope at hydro res (needed by SPI/TWI), WBT-friendly nodata
    # hydro_slope = p("_slope_hydro.tif")
    # local_derivatives(hydro_dem, out_slope=hydro_slope, nodata_out=NODATA)

    # # 6b. Breach + flow accumulation (WBT, on small DEM)
    # wbt = whitebox.WhiteboxTools()
    # wbt.set_verbose_mode(True)
    # wbt.set_compress_rasters(False)

    # breached = p("_dem_breached.tif")
    # wbt.breach_depressions_least_cost(hydro_dem, breached, dist=5, fill=True)
    # check(breached, "BreachDepressions")

    # wbt.d8_flow_accumulation(breached, p("flow_accumulation.tif"), out_type="cells")
    # check(p("flow_accumulation.tif"), "D8FlowAccumulation")

    # # 7-8. SPI / TWI
    # print("[7/9] SPI …")
    # wbt.stream_power_index(p("flow_accumulation.tif"), hydro_slope, p("spi.tif"))
    # check(p("spi.tif"), "SPI")

    # print("[8/9] TWI …")
    # wbt.wetness_index(p("flow_accumulation.tif"), hydro_slope, p("twi.tif"))
    # check(p("twi.tif"), "TWI")

    # # 9. Drainage density (at hydro res, with PROPER metre conversion)
    # print("[9/9] Drainage density …")
    # fa, fa_prof = read_band(p("flow_accumulation.tif"))
    # cs_m = hydro_res_deg * M_PER_DEG * np.cos(np.radians(mean_lat))  # deg -> metres

    # streams = np.where(np.isfinite(fa) & (fa >= stream_threshold), 1.0, 0.0)
    # Wd = drainage_window_px
    # stream_sum  = uniform_filter(streams, size=Wd, mode="reflect") * float(Wd * Wd)
    # window_area = (Wd * cs_m) ** 2
    # dd = (stream_sum * cs_m) / window_area
    # dd[~np.isfinite(fa)] = np.nan
    # dd_prof = out_profile(fa_prof, nodata=np.nan)
    # with rasterio.open(p("drainage_density.tif"), "w", **dd_prof) as dst:
    #     dst.write(dd.astype(np.float32), 1)

    # # cleanup temps
    # for t in (breached, hydro_dem, hydro_slope):
    #     if os.path.exists(t):
    #         os.remove(t)

    # files = ["elevation.tif", "slope.tif", "aspect.tif", "plan_curvature.tif",
    #          "profile_curvature.tif", "flow_accumulation.tif", "spi.tif",
    #          "twi.tif", "drainage_density.tif"]

    files = ["elevation.tif", "slope.tif"]

    print(f"\n✅  Done! {len(files)} layers -> {os.path.abspath(output_dir)}")
    for f in files:
        print(f"   {f:<32s}  {os.path.getsize(p(f))/1024/1024:>9.1f} MB")


if __name__ == "__main__":
    main()