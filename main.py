import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import BoxLeastSquares


# ============================================================
# EXOPLANET ANALYSIS FUNCTION
# ============================================================

def analyze_exoplanet(star_name):

    print("\n==========================================")
    print(f"Analyzing: {star_name}")
    print("==========================================")

    # 1. in this section tess data is being searched on their databse

    print("\nSearching for TESS SPOC data...")

    search = lk.search_lightcurve(
        star_name,
        mission="TESS",
        author="SPOC"
    )

    print(search)
    print(f"\nFound {len(search)} observations.")

    # --------------------------------------------------------
    # 2. DOWNLOAD AND COMBINE OBSERVATIONS
    # --------------------------------------------------------

    print("\nDownloading TESS observations...")

    lightcurves = search.download_all()

    print(f"Downloaded {len(lightcurves)} observations.")

    print("Combining observations...")

    lightcurve = lightcurves.stitch()

    print("All observations combined!")

    # --------------------------------------------------------
    # 3. CLEAN DATA
    # --------------------------------------------------------

    print("\nCleaning data...")

    clean_lc = lightcurve.remove_nans()

    clean_lc = clean_lc.remove_outliers(
        sigma=5
    )

    # --------------------------------------------------------
    # 4. NORMALIZE
    # --------------------------------------------------------

    print("Normalizing data...")

    normalized_lc = clean_lc.normalize()

    # --------------------------------------------------------
    # 5. DETREND
    # --------------------------------------------------------

    print("Detrending light curve...")

    detrended_lc = normalized_lc.flatten(
        window_length=101
    )

    # --------------------------------------------------------
    # 6. PREPARE DATA FOR BLS
    # --------------------------------------------------------

    time = detrended_lc.time.value
    flux = detrended_lc.flux.value

    mask = np.isfinite(time) & np.isfinite(flux)

    time = time[mask]
    flux = flux[mask]

    # --------------------------------------------------------
    # 7. BLS PERIOD SEARCH
    # --------------------------------------------------------

    print("\nRunning BLS period search...")

    bls = BoxLeastSquares(
        time,
        flux
    )

    periods = np.linspace(
        1.0,
        10.0,
        10000
    )

    durations = np.linspace(
        0.03,
        0.30,
        30
    )

    results = bls.power(
        periods,
        durations
    )

    # --------------------------------------------------------
    # 8. FIND BEST SIGNAL
    # --------------------------------------------------------

    best_index = np.argmax(
        results.power
    )

    best_period = results.period[best_index]

    best_duration = results.duration[best_index]

    best_power = results.power[best_index]

    best_transit_time = results.transit_time[best_index]

    # --------------------------------------------------------
    # 9. PRINT RESULTS
    # --------------------------------------------------------

    print("\n========== BLS RESULTS ==========")

    print(
        f"star_name: {star_name}"
    )

    print(
        f"Best period: {best_period:.6f} days"
    )

    print(
        f"Best duration: {best_duration:.6f} days"
    )

    print(
        f"BLS power: {best_power:.6f}"
    )

    print(
        f"Transit epoch: {best_transit_time:.6f}"
    )

    print("=================================")

    # --------------------------------------------------------
    # 10. PHASE FOLD
    # --------------------------------------------------------

    print("\nPhase-folding light curve...")

    folded_lc = detrended_lc.fold(
        period=best_period,
        epoch_time=best_transit_time,
        normalize_phase=True
    )

    # --------------------------------------------------------
    # 11. BIN DATA
    # --------------------------------------------------------

    print("Binning phase-folded data...")

    binned_lc = folded_lc.bin(
        bins=150
    )

    # --------------------------------------------------------
    # 12. GRAPHS
    # --------------------------------------------------------

    plt.figure(
        f"{star_name} - Raw TESS Data",
        figsize=(12, 5)
    )

    plt.plot(
        lightcurve.time.value,
        lightcurve.flux.value,
        linewidth=0.5
    )

    plt.xlabel("Time - 2457000 [BTJD days]")
    plt.ylabel("Flux")

    plt.title(
        f"{star_name} — Raw TESS Light Curve"
    )

    plt.grid()

    plt.show(block=False)


    # --------------------------------------------------------

    plt.figure(
        f"{star_name} - Cleaned Data",
        figsize=(12, 5)
    )

    plt.plot(
        normalized_lc.time.value,
        normalized_lc.flux.value,
        linewidth=0.5
    )

    plt.xlabel("Time - 2457000 [BTJD days]")
    plt.ylabel("Normalized Flux")

    plt.title(
        f"{star_name} — Cleaned & Normalized"
    )

    plt.grid()

    plt.show(block=False)


    # --------------------------------------------------------

    plt.figure(
        f"{star_name} - Detrended Data",
        figsize=(12, 5)
    )

    plt.plot(
        detrended_lc.time.value,
        detrended_lc.flux.value,
        linewidth=0.5
    )

    plt.xlabel("Time - 2457000 [BTJD days]")
    plt.ylabel("Normalized Flux")

    plt.title(
        f"{star_name} — Detrended TESS Light Curve"
    )

    plt.grid()

    plt.show(block=False)


    # --------------------------------------------------------

    plt.figure(
        f"{star_name} - Phase Folded",
        figsize=(12, 5)
    )

    plt.scatter(
        folded_lc.phase.value,
        folded_lc.flux.value,
        s=2
    )

    plt.xlabel("Orbital Phase")
    plt.ylabel("Normalized Flux")

    plt.title(
        f"{star_name} — Phase-Folded Transit\n"
        f"Detected Period = {best_period:.6f} days"
    )

    plt.xlim(
        -0.1,
        0.1
    )

    plt.grid()

    plt.show(block=False)


    # --------------------------------------------------------

    plt.figure(
        f"{star_name} - Binned Transit",
        figsize=(12, 5)
    )

    plt.scatter(
        binned_lc.phase.value,
        binned_lc.flux.value,
        s=12
    )

    plt.xlabel("Orbital Phase")
    plt.ylabel("Normalized Flux")

    plt.title(
        f"{star_name} — Binned Phase-Folded Transit\n"
        f"Period = {best_period:.6f} days"
    )

    plt.xlim(
        -0.1,
        0.1
    )

    plt.grid()

    plt.show()


# ============================================================
# RUN THE ANALYSIS
# ============================================================

analyze_exoplanet("HD 209458")