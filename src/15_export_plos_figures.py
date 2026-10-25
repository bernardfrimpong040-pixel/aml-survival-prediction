"""
Stage 9 - Export the figures for the PLOS ONE submission.

PLOS ONE does not want figures inside the manuscript PDF. Each figure is uploaded
as its own file, named Fig1.tif, Fig2.tif, ... (and S1_Fig.tif, ... for supporting
figures). PLOS asks for:
  - TIFF (or EPS) format, RGB colour, LZW compression
  - 300 to 600 dots per inch (dpi)
  - width between 2.63 and 7.5 inches, height at most 8.75 inches

This script reads YOUR figures from figures/ (made by scripts 07 to 13 and 16), converts
each one to that format and saves it in paper_plos/figures_upload/.

Run it from the project folder:  python src/15_export_plos_figures.py
"""

from pathlib import Path

from PIL import Image

import config

# ------------------------------------------------------------------
# 1. Which figure becomes which PLOS file.
#    The order matches the order in which the manuscript cites them.
# ------------------------------------------------------------------
FIGURE_MAP = {
    "Fig1.tif": "fig25_ipssm.png",                 # survival by IPSS-M category
    "Fig2.tif": "fig26_paired_ci.png",             # paired differences with CIs
    "Fig3.tif": "fig15_hazard_ratios.png",         # hazard ratios
    "Fig4.tif": "fig23_permutation_importance.png",
    "Fig5.tif": "fig28_calibration.png",
    "Fig6.tif": "fig24_leaderboard.png",           # CV against public score
    "Fig7.tif": "fig29_ipssm_gap_by_centre.png",   # IPSS-M gap by centre (script 16)
    "S1_Fig.tif": "fig14_cox_cv.png",              # Cox models fold by fold
    "S2_Fig.tif": "fig19_tree_cv.png",             # tree models fold by fold
    "S3_Fig.tif": "fig16_schoenfeld.png",          # time-varying effects
    "S4_Fig.tif": "fig27_time_auc.png",            # time-dependent AUC
}

TARGET_DPI = 300        # PLOS minimum, and the usual choice
MAX_WIDTH_IN = 7.5      # PLOS maximum width in inches
MAX_HEIGHT_IN = 8.75    # PLOS maximum height in inches
MIN_WIDTH_IN = 2.63     # PLOS minimum width in inches

out_folder = config.ROOT / "paper_plos" / "figures_upload"
out_folder.mkdir(parents=True, exist_ok=True)

print(f"Saving PLOS figures to {out_folder}\n")

for plos_name, source_name in FIGURE_MAP.items():
    source = config.FIGURES / source_name
    if not source.exists():
        print(f"  MISSING {source_name}: run the script that makes it first.")
        continue

    image = Image.open(source)

    # ------------------------------------------------------------------
    # 2. Size in inches as it was saved.
    #    Our scripts save at 200 dpi, so a 2100-pixel-wide figure is 10.5 inches.
    # ------------------------------------------------------------------
    saved_dpi = image.info.get("dpi", (200, 200))[0]
    width_in = image.width / saved_dpi
    height_in = image.height / saved_dpi

    # ------------------------------------------------------------------
    # 3. Shrink (never enlarge) so it fits PLOS's maximum width and height.
    #    The same factor is used for both directions so nothing is stretched.
    # ------------------------------------------------------------------
    scale = min(1.0, MAX_WIDTH_IN / width_in, MAX_HEIGHT_IN / height_in)
    new_width_in = width_in * scale
    new_height_in = height_in * scale

    # Number of pixels needed at 300 dpi for that printed size.
    new_width_px = round(new_width_in * TARGET_DPI)
    new_height_px = round(new_height_in * TARGET_DPI)

    # ------------------------------------------------------------------
    # 4. PLOS wants RGB with no transparency: put the figure on a white page.
    # ------------------------------------------------------------------
    image = image.convert("RGBA")
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(white, image).convert("RGB")

    # LANCZOS is a high-quality resampling filter for resizing images.
    image = image.resize((new_width_px, new_height_px), Image.LANCZOS)

    # ------------------------------------------------------------------
    # 5. Save as LZW-compressed TIFF with the dpi written into the file.
    # ------------------------------------------------------------------
    image.save(out_folder / plos_name, compression="tiff_lzw",
               dpi=(TARGET_DPI, TARGET_DPI))

    # Text shrinks by the same factor; PLOS asks for 8-12 pt text at final size.
    font_pt = 10 * scale
    warning = "" if new_width_in >= MIN_WIDTH_IN else "  (narrower than 2.63 in!)"
    print(f"  {plos_name:11s} <- {source_name:34s} "
          f"{new_width_in:4.2f} x {new_height_in:4.2f} in, "
          f"{new_width_px} x {new_height_px} px, text about {font_pt:.1f} pt{warning}")

print("\nDone. Before uploading, you can check each file with PLOS's free PACE tool:")
print("https://pacev2.apexcovantage.com")
