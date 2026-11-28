# File name: feed_compare_hfss.py
# Authors: Stav Shulman, Philippe Logoz, Cédric Zaugg
# File created: 27.11.2026

"""
Goal of this script: compare two microstrip -> slotline feeds of the Vivaldi antenna in Ansys HFSS.

    "current"  the feed of vivaldi_hfss.py: 50-ohm line, linear taper to ~100 ohm, overshoot, tilted radial stub
    "stepped"  the feed of Schéma_stripline_slotline.pdf, dimensioned by code_python/calcul.py:
               N-section Chebyshev transformer 50 ohm -> Zom = n^2 Zos (widths W1..Wn, lengths L1..Ln),
               the narrowest section Wn crosses the slot, a 90 deg radial stub of radius lambda_m/4 starts
               at the far edge of the slot, and the circular cavity has radius lambda_s/4.

Usage
    python feed_compare_hfss.py                 build both cut models (no flare) in one HFSS project
    python feed_compare_hfss.py --solve         build, solve, write the reports and a comparison plot
    python feed_compare_hfss.py --preview       just draw both outlines with matplotlib, does not open HFSS
    python feed_compare_hfss.py --full ...      same, but with the exponential flare (whole antenna, slower)

Cut model (default, much faster): the board stops L_OUT mm after the microstrip and the slotline runs straight
to the board end, where it is terminated by a second lumped port with the slotline impedance Zos. This gives a
2-port that only contains the transition:
    S11      how well the feed is matched at the 50-ohm connector
    S21      how much power actually gets into the slotline (what would feed the flare)
    1-|S11|^2-|S21|^2   power radiated or dissipated by the transition itself (stub, cavity, line losses)

Layout: all dimensions in mm, same axes as vivaldi_hfss.py (slot along +x, microstrip along y, port at -y).

Important notes
- Be sure to be on EPFL's VPN before running the scripts! Otherwise the program crashes
- Same Python recommendation as vivaldi_hfss.py (3.12)
"""

# --------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------

import argparse
import contextlib
import io
import math
import os
import runpy

import vivaldi_hfss as V        # current design: dimensions and outlines are taken from there

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------------------
# Parameters of the comparison
# --------------------------------------------------------------------------------------
L_OUT = 15.0                # cut model: board end (and slot port) this far past the microstrip centre line
STUB_TILT_NEW = 0.0         # stepped feed: stub tilt towards +x, degrees (0 = symmetric, as on the drawing)
FEED_OFFSET_NEW = None      # stepped feed: cavity edge -> microstrip centre line. None = smallest value that
                            # keeps the stub STUB_CLEARANCE away from the cavity (computed automatically)
N_STUB = 24                 # segments of the stub arc

PROJECT = "Vivaldi_feed_compare"


# --------------------------------------------------------------------------------------
# Values from code_python/calcul.py (SI units there, converted to mm here)
# --------------------------------------------------------------------------------------
def load_calcul():
    with contextlib.redirect_stdout(io.StringIO()):     # calcul.py prints its table on import
        return runpy.run_path(os.path.join(HERE, "code_python", "calcul.py"))


C = load_calcul()
ER, H = C["er"], C["h"] * 1e3
W_SLOT_NEW = C["Wsl"] * 1e3                     # slotline width
D_CAV_NEW = 2.0 * (C["Lamdas"] / 4.0) * 1e3     # cavity radius lambda_s / 4
R_STUB_NEW = (C["lamdam"] / 4.0) * 1e3          # stub radius lambda_m / 4 (measured from the slot edge)
W_SECT = [w * 1e3 for w in C["Wn"]]             # W1 (next to 50 ohm) ... Wn (crosses the slot)
L_SECT = [l * 1e3 for l in C["Ln"]]
W_50 = C["CalculWms"](ER, 50.0, C["h"]) * 1e3   # 50-ohm feed line between the port and section 1


def slot_impedance(w_slot_mm):
    # Slotline impedance (Janaswamy & Schaubert, as in calcul.py) at the geometric centre frequency
    with contextlib.redirect_stdout(io.StringIO()):
        return C["calculZos"](ER, w_slot_mm * 1e-3, C["lamda0s"], C["h"])


# --------------------------------------------------------------------------------------
# Geometry helpers (shared by both designs)
# --------------------------------------------------------------------------------------
def radial_stub(x_feed, y_mouth, w_strip, radius, angle, tilt, n=N_STUB):
    """Radial stub fanning towards +y. Its straight edges cross the strip edges at y = y_mouth, and
    `radius` is measured from there; the apex is a little lower so the stub overlaps the strip."""
    half, tilt = math.radians(angle / 2.0), math.radians(tilt)
    back = (w_strip / 2.0) / math.tan(half)
    apex = [x_feed, y_mouth - back]
    r = radius + back
    pts = [apex]
    for i in range(n + 1):
        a = tilt - half + 2.0 * half * i / n           # angle from the +y axis
        pts.append([apex[0] + r * math.sin(a), apex[1] + r * math.cos(a)])
    return pts


def taper_half_width(x, x_throat, w_slot):
    # Same exponential flare as vivaldi_hfss.py, but for any throat position / slot width
    x1, y1 = x_throat, w_slot / 2.0
    x2, y2 = V.L_BOARD, V.W_APERTURE / 2.0
    e1, e2 = math.exp(V.R_TAPER * x1), math.exp(V.R_TAPER * x2)
    return (y2 - y1) / (e2 - e1) * math.exp(V.R_TAPER * x) + (y1 * e2 - y2 * e1) / (e2 - e1)


def flare_outline(x_throat, w_slot):
    xs = [x_throat + (V.L_BOARD - x_throat) * i / (V.N_TAPER - 1) for i in range(V.N_TAPER)]
    return [[x, taper_half_width(x, x_throat, w_slot)] for x in xs] + \
           [[x, -taper_half_width(x, x_throat, w_slot)] for x in reversed(xs)]


def stub_gaps(stub, y_strip_end, x_cav_c, d_cav, slot_half):
    """Smallest gaps between the stub (beyond the strip end) and the cavity / the slot.
    slot_half(x) gives the half-width of the slot opening at x."""
    apex, arc = stub[0], stub[1:]
    gap_cav = gap_slot = float("inf")
    for px, py in arc:
        for k in range(1, 41):
            x = apex[0] + (px - apex[0]) * k / 40.0
            y = apex[1] + (py - apex[1]) * k / 40.0
            if y <= y_strip_end:
                continue
            gap_cav = min(gap_cav, math.hypot(x - x_cav_c, y) - d_cav / 2.0)
            # the stub of the drawing starts on the slot edge: only check the slot gap 1 mm beyond the strip end
            if x >= x_cav_c and y > y_strip_end + 1.0:
                gap_slot = min(gap_slot, abs(y) - slot_half(x))
    return gap_cav, gap_slot


# --------------------------------------------------------------------------------------
# The two designs
# --------------------------------------------------------------------------------------
def design_current(full):
    """Feed of vivaldi_hfss.py, unchanged."""
    d = dict(name="Feed_current", w_slot=V.W_SLOT, d_cav=V.D_CAV, x_cav=V.X_CAV, x_feed=V.X_FEED,
             x_throat=V.X_THROAT, metal=[V.feed_outline(), V.stub_outline()], y_strip_end=V.MS_OVERSHOOT,
             w_port=V.W_MS)
    return _finish(d, full)


def stepped_strip(x_feed, y_mouth):
    """Staircase polygon: 50-ohm line from the board edge, then sections 1..n, ending at y_mouth."""
    y = y_mouth - sum(L_SECT)                       # bottom of section 1
    right = [[x_feed + W_50 / 2, -V.W_BOARD / 2], [x_feed + W_50 / 2, y]]
    for w, l in zip(W_SECT, L_SECT):
        right += [[x_feed + w / 2, y], [x_feed + w / 2, y + l]]
        y += l
    left = [[2 * x_feed - px, py] for px, py in reversed(right)]
    return right + left


def design_stepped(full):
    """Feed of Schéma_stripline_slotline.pdf with the dimensions of calcul.py."""
    y_mouth = W_SLOT_NEW / 2.0                       # stub and section n start at the far slot edge
    x_cav = V.X_CAV

    def make(offset):
        x_feed = x_cav + D_CAV_NEW + offset
        stub = radial_stub(x_feed, y_mouth, W_SECT[-1], R_STUB_NEW, 90.0, STUB_TILT_NEW)
        return x_feed, stub

    offset = FEED_OFFSET_NEW
    if offset is None:                               # walk away from the cavity until the stub clears it
        offset = W_SECT[-1] / 2.0 + 0.5
        while True:
            x_feed, stub = make(offset)
            gap, _ = stub_gaps(stub, y_mouth, x_cav + D_CAV_NEW / 2, D_CAV_NEW, lambda x: W_SLOT_NEW / 2)
            if gap >= V.STUB_CLEARANCE:
                break
            offset = round(offset + 0.1, 3)
    x_feed, stub = make(offset)
    # keep the same "throat - feed" distance as the current design
    d = dict(name="Feed_stepped", w_slot=W_SLOT_NEW, d_cav=D_CAV_NEW, x_cav=x_cav, x_feed=x_feed,
             x_throat=x_feed + (V.X_THROAT - V.X_FEED), metal=[stepped_strip(x_feed, y_mouth), stub],
             y_strip_end=y_mouth, feed_offset=offset, w_port=W_50)
    return _finish(d, full)


def _finish(d, full):
    d["x_cav_c"] = d["x_cav"] + d["d_cav"] / 2.0
    d["full"] = full
    d["x_end"] = V.L_BOARD if full else d["x_feed"] + L_OUT
    d["z_slot"] = slot_impedance(d["w_slot"])
    if full:
        d["slot_half"] = lambda x: (taper_half_width(x, d["x_throat"], d["w_slot"]) if x > d["x_throat"]
                                    else d["w_slot"] / 2.0)
    else:
        d["slot_half"] = lambda x: d["w_slot"] / 2.0
    if not full:
        d["name"] += "_cut"
    return d


def check_design(d):
    gap_cav, gap_slot = stub_gaps(d["metal"][1], d["y_strip_end"], d["x_cav_c"], d["d_cav"], d["slot_half"])
    assert gap_cav >= V.STUB_CLEARANCE - 1e-9, f"{d['name']}: stub is {gap_cav:.2f} mm from the cavity."
    assert gap_slot >= 0.5, f"{d['name']}: stub is {gap_slot:.2f} mm from the slot / flare."
    xs = [p[0] for poly in d["metal"] for p in poly]
    assert max(xs) < d["x_end"] - 1.0, f"{d['name']}: stub runs past the end of the board (raise L_OUT)."
    assert min(p[0] for p in d["metal"][0]) > d["x_cav"] + d["d_cav"], \
        f"{d['name']}: microstrip runs over the cavity."
    if d["full"]:
        assert d["x_throat"] < V.L_BOARD - 20.0, f"{d['name']}: almost no flare left on the board."


def check_parameters(designs):
    assert abs(ER - V.EPS_R) < 1e-9 and abs(H - V.T_SUB) < 1e-9, \
        "calcul.py and vivaldi_hfss.py use different substrates (er / h)."
    lead = V.W_BOARD / 2.0 - sum(L_SECT) + W_SLOT_NEW / 2.0
    assert lead > 0, f"The {len(L_SECT)}-section transformer ({sum(L_SECT):.1f} mm) is longer than half the board."
    if lead < 2.0:
        print(f"[warning] only {lead:.2f} mm of 50-ohm line between the port and the transformer")
    for d in designs:
        check_design(d)


def describe(designs):
    print(f"calcul.py: er = {ER}, h = {H} mm, f_s = {299.792458 / C['lamda0s'] / 1e3:.3f} GHz, "
          f"Zos = {C['Zos']:.1f} ohm, n = {C['n']:.3f}, Zom = {C['Zom']:.1f} ohm, N = {C['N']}")
    print(f"  cavity diameter 2*lambda_s/4 = {D_CAV_NEW:.2f} mm, stub radius lambda_m/4 = {R_STUB_NEW:.2f} mm, "
          f"50-ohm width = {W_50:.3f} mm")
    for i, (z, w, l) in enumerate(zip(C["Zn"], W_SECT, L_SECT), 1):
        print(f"  section {i}: Z = {z:6.2f} ohm   W = {w:.3f} mm   L = {l:.3f} mm")
    for d in designs:
        extra = f", feed offset {d['feed_offset']:.2f} mm" if "feed_offset" in d else ""
        print(f"{d['name']}: slot {d['w_slot']} mm (Zos = {d['z_slot']:.1f} ohm), cavity {d['d_cav']:.2f} mm, "
              f"x_feed = {d['x_feed']:.2f} mm, board x = 0..{d['x_end']:.1f} mm{extra}")


# --------------------------------------------------------------------------------------
# Matplotlib preview
# --------------------------------------------------------------------------------------
def preview(designs, path="feed_compare_preview.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    fig, axes = plt.subplots(1, len(designs), figsize=(7 * len(designs), 7), squeeze=False)
    for ax, d in zip(axes[0], designs):
        ax.add_patch(Rectangle((0, -V.W_BOARD / 2), d["x_end"], V.W_BOARD, fc="#c9a25a", ec="k",
                               label="ground metal (top)"))
        slot = dict(fc="white", ec="k", lw=0.6)
        ax.add_patch(Circle((d["x_cav_c"], 0), d["d_cav"] / 2, **slot))
        x_slot_end = d["x_throat"] if d["full"] else d["x_end"]
        ax.add_patch(Rectangle((d["x_cav_c"], -d["w_slot"] / 2), x_slot_end - d["x_cav_c"], d["w_slot"],
                               fc="white", ec="none"))
        if d["full"]:
            ax.add_patch(Polygon(flare_outline(d["x_throat"], d["w_slot"]), **slot))
        else:
            ax.plot([d["x_end"]], [0], "g<", ms=10, label=f"slot port ({d['z_slot']:.0f} ohm)")
        for i, poly in enumerate(d["metal"]):
            ax.add_patch(Polygon(poly, fc="#1f77b4", ec="none", alpha=0.6,
                                 label="microstrip + stub (bottom)" if i == 0 else None))
        ax.plot([d["x_feed"]], [-V.W_BOARD / 2], "rv", ms=10, label="50-ohm port")
        ax.set_xlim(-5, d["x_end"] + 5)
        ax.set_ylim(-V.W_BOARD / 2 - 5, V.W_BOARD / 2 + 5)
        ax.set_aspect("equal")
        ax.set_title(d["name"])
        ax.set_xlabel("x [mm]  (radiation ->)")
        ax.set_ylabel("y [mm]")
        ax.legend(loc="upper left", fontsize=8)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"Preview written to {path}")


# --------------------------------------------------------------------------------------
# HFSS model
# --------------------------------------------------------------------------------------
def build_design(hfss, d):
    hfss.modeler.model_units = "mm"
    m = hfss.modeler

    def polygon(xy, z, name):
        return m.create_polyline([[x, y, z] for x, y in xy], cover_surface=True, close_surface=True, name=name)

    # ---- substrate (the material lives in the project, so only create it once) -------
    if not hfss.materials.exists_material("Substrate_FR4"):
        mat = hfss.materials.add_material("Substrate_FR4")
        mat.permittivity = V.EPS_R
        mat.dielectric_loss_tangent = V.TAN_D
    x_end = d["x_end"]
    m.create_box([0, -V.W_BOARD / 2, 0], [x_end, V.W_BOARD, V.T_SUB], name="Substrate", material="Substrate_FR4")

    # ---- top metal: ground minus (cavity + slotline [+ flare]) -----------------------
    ground = m.create_rectangle("XY", [0, -V.W_BOARD / 2, V.T_SUB], [x_end, V.W_BOARD], name="Ground")
    cavity = m.create_circle("XY", [d["x_cav_c"], 0, V.T_SUB], d["d_cav"] / 2, name="Cavity")
    x_slot_end = d["x_throat"] if d["full"] else x_end
    slotline = m.create_rectangle("XY", [d["x_cav_c"], -d["w_slot"] / 2, V.T_SUB],
                                  [x_slot_end - d["x_cav_c"], d["w_slot"]], name="Slotline")
    openings = [cavity, slotline]
    if d["full"]:
        openings.append(polygon(flare_outline(d["x_throat"], d["w_slot"]), V.T_SUB, "Flare"))
    m.unite(openings)
    m.subtract(ground, cavity, keep_originals=False)

    # ---- bottom metal: microstrip + stub -----------------------------------------------
    parts = [polygon(poly, 0, f"Feed{i}") for i, poly in enumerate(d["metal"])]
    m.unite(parts)
    strip = parts[0]
    ground.color, strip.color = (200, 160, 60), (30, 110, 180)
    if V.USE_COPPER:
        hfss.assign_finite_conductivity([ground.name, strip.name], material="copper", name="Metal")
    else:
        hfss.assign_perfecte_to_sheets([ground.name, strip.name], name="Metal")

    # ---- port 1: 50-ohm lumped port at the board edge (strip z = 0 -> ground z = h) ----
    y_edge = -V.W_BOARD / 2
    x0, x1 = d["x_feed"] - d["w_port"] / 2, d["x_feed"] + d["w_port"] / 2
    p1 = m.create_polyline([[x0, y_edge, 0], [x1, y_edge, 0], [x1, y_edge, V.T_SUB], [x0, y_edge, V.T_SUB]],
                           cover_surface=True, close_surface=True, name="PortSheet1")
    hfss.lumped_port(assignment=p1.name, create_port_sheet=False,
                     integration_line=[[d["x_feed"], y_edge, 0], [d["x_feed"], y_edge, V.T_SUB]],
                     impedance=50, name="P1", renormalize=True)

    # ---- port 2 (cut model only): across the slot gap at the board end, Zos ----------
    if not d["full"]:
        w = d["w_slot"]
        p2 = m.create_rectangle("XY", [x_end - w, -w / 2, V.T_SUB], [w, w], name="PortSheet2")
        hfss.lumped_port(assignment=p2.name, create_port_sheet=False,
                         integration_line=[[x_end - w / 2, -w / 2, V.T_SUB], [x_end - w / 2, w / 2, V.T_SUB]],
                         impedance=round(d["z_slot"], 2), name="P2", renormalize=True)

    # ---- radiation boundary and analysis (same settings as vivaldi_hfss.py) -----------
    hfss.create_open_region(frequency=f"{V.F_LO}GHz", boundary="Radiation")
    setup = hfss.create_setup("Setup1", Frequency=f"{V.F_HI}GHz", MaximumPasses=V.MAX_PASSES,
                              MaxDeltaS=V.MAX_DELTA_S, MinimumConvergedPasses=2)
    try:
        ok = setup.enable_adaptive_setup_broadband(f"{V.F_LO}GHz", f"{V.F_HI}GHz", V.MAX_PASSES, V.MAX_DELTA_S)
    except Exception as err:                                   # noqa: BLE001
        ok = False
        print(f"[warning] broadband adaptive not set: {err}")
    if not ok:
        print(f"[warning] falling back to single-frequency adaptive meshing at {V.F_HI} GHz")
    hfss.create_linear_count_sweep(setup.name, "GHz", V.F_LO - 1.0, V.F_HI + 1.0,
                                   num_of_freq_points=V.SWEEP_POINTS, name="Sweep", sweep_type="Interpolating",
                                   save_fields=False)
    m.fit_all()
    return setup.name


def solve_design(hfss, d, setup_name):
    """Solve, add the reports to the design, export a Touchstone file and return the curves."""
    hfss.analyze(setup=setup_name)
    post = hfss.post
    sweep = f"{setup_name} : Sweep"
    out_dir = os.path.dirname(hfss.project_file)
    exprs = ["dB(S(P1,P1))", "re(Z(P1,P1))", "im(Z(P1,P1))"]
    if not d["full"]:
        exprs += ["dB(S(P2,P1))", "1-mag(S(P1,P1))^2-mag(S(P2,P1))^2"]

    reports = [("S-parameters", [e for e in exprs if e.startswith("dB")]),
               ("Input impedance", ["re(Z(P1,P1))", "im(Z(P1,P1))"])]
    if not d["full"]:
        reports.append(("Radiated + lost power fraction", ["1-mag(S(P1,P1))^2-mag(S(P2,P1))^2"]))
    for name, ex in reports:
        try:
            post.create_report(ex, sweep, plot_name=name)
        except Exception as err:                               # noqa: BLE001
            print(f"[failed] report {name}: {err}")
    try:
        post.create_report("S(P1,P1)", sweep, plot_type="Smith Chart", plot_name="Smith chart")
        hfss.export_touchstone(setup_name, "Sweep", os.path.join(out_dir, f"{d['name']}.s{1 if d['full'] else 2}p"))
    except Exception as err:                                   # noqa: BLE001
        print(f"[failed] Smith chart / Touchstone: {err}")

    curves = {}
    try:
        data = post.get_solution_data(expressions=exprs, setup_sweep_name=sweep)
        for e in exprs:
            curves[e] = data.get_expression_data(e, formula="real")
    except Exception as err:                                   # noqa: BLE001
        print(f"[failed] reading results of {d['name']}: {err}")
    return curves


def compare_plot(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    two_port = any("dB(S(P2,P1))" in c for c in results.values())
    n = 3 if two_port else 2
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.2 * n), sharex=True)
    for name, c in results.items():
        if "dB(S(P1,P1))" not in c:
            continue
        f, s11 = c["dB(S(P1,P1))"]
        (line,) = axes[0].plot(f, s11, label=f"{name}  S11")
        if "dB(S(P2,P1))" in c:
            axes[0].plot(*c["dB(S(P2,P1))"], "--", color=line.get_color(), label=f"{name}  S21")
            axes[2].plot(*c["1-mag(S(P1,P1))^2-mag(S(P2,P1))^2"], color=line.get_color(), label=name)
        axes[1].plot(*c["re(Z(P1,P1))"], color=line.get_color(), label=f"{name}  Re")
        axes[1].plot(*c["im(Z(P1,P1))"], ":", color=line.get_color(), label=f"{name}  Im")
    axes[0].axhline(-10, color="k", lw=0.8, ls="--")
    axes[0].set_ylabel("|S| [dB]")
    axes[1].axhline(50, color="k", lw=0.8, ls="--")
    axes[1].set_ylabel("Z_in [ohm]")
    if two_port:
        axes[2].set_ylabel("1 - |S11|² - |S21|²")
    for ax in axes:
        ax.axvspan(V.F_LO, V.F_HI, color="0.9", zorder=0)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Frequency [GHz]")
    path = os.path.join(out_dir, "feed_compare.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"Comparison plot written to {path}")


def build_all(designs, solve=False, non_graphical=False, version=None):
    from ansys.aedt.core import Hfss

    hfss = None
    results = {}
    for d in designs:
        if hfss is None:
            hfss = Hfss(project=PROJECT, design=d["name"], solution_type="Modal", version=version,
                        non_graphical=non_graphical, new_desktop=True)
        else:
            hfss.insert_design(d["name"])          # same project, the app now points to the new design
        setup_name = build_design(hfss, d)
        hfss.save_project()
        if solve:
            results[d["name"]] = solve_design(hfss, d, setup_name)
            hfss.save_project()
    if solve:
        compare_plot(results, os.path.dirname(hfss.project_file))
    print(f"Project saved to {hfss.project_file}")
    return hfss


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true", help="draw both outlines only, without HFSS")
    ap.add_argument("--solve", action="store_true", help="run the simulations after building")
    ap.add_argument("--full", action="store_true", help="include the flare (whole antenna) instead of the cut model")
    ap.add_argument("--only", choices=["current", "stepped"], help="build only one of the two designs")
    ap.add_argument("--non-graphical", action="store_true", help="run HFSS without the GUI")
    ap.add_argument("--version", default=None, help='AEDT version, e.g. "2025.1" (default: newest installed)')
    args = ap.parse_args()

    designs = [design_current(args.full), design_stepped(args.full)]
    if args.only:
        designs = [d for d in designs if d["name"].startswith(f"Feed_{args.only}")]
    check_parameters(designs)
    describe(designs)
    if args.preview:
        preview(designs)
    else:
        app = build_all(designs, solve=args.solve, non_graphical=args.non_graphical, version=args.version)
        if args.non_graphical:
            app.release_desktop()
