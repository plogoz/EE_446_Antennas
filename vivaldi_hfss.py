# File name: vivaldi_hfss.py
# Authors: Stav Shulman, Philippe Logoz, Cédric Zaugg
# File created: 16.11.2026

"""
Goal of this script: build a microstrip-fed Vivaldi antenna in Ansys HFSS
Usage
    python vivaldi_hfss.py              build the model in HFSS
    python vivaldi_hfss.py --solve      build, solve, and plot S11
    python vivaldi_hfss.py --preview    just draw the outline with matplotlib, does not open HFSS
    python vivaldi_hfss.py --slot-port --solve
                                        same antenna plus a second lumped port across the slot, at the same spot and
                                        with the same impedance as the slot port of the cut model in
                                        feed_compare_hfss.py (separate design "Vivaldi_microstrip_fed_P2")

Slot port note: an HFSS port is always a matched load (even when it is not excited), so P2 sits in parallel with
the rest of the flare and absorbs part of the power. Comparing S11 / S21 of this design with the cut model tells
whether the flare changes what the feed sees; comparing its S11 with the design without P2 shows how much the
port itself disturbs the antenna. Far fields are computed with P1 excited only.

Layout: all dimensions in mm!

Flare geometry: y(x) = c1 *exp(R * x) + c2
    c1, c2: fixed by the throat and aperture
    R: opening rate

Important notes
- Be sure to be on EPFL's VPN before running the scripts! Otherwise the program crashes
- I had issues running the script on Python 3.14, so I recommend using Python 3.12,
"""

# --------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------

import argparse
import math
import os

# --------------------------------------------------------------------------------------
# Design parameters
# --------------------------------------------------------------------------------------
F_LO, F_HI = 2.0, 7.0       # Interested in the 2-7 GHz frequency range

# Substrate (FR4)
EPS_R = 4.4
TAN_D = 0.02                # Unsure about this value
T_SUB = 0.8                 # Substrate thickness: we can choose either 0.8mm or 1.6mm
                            # (Yngvesson, 1989) -> we choose 0.8mm (see notes on Notion)
"""
The following values have been chosen randomly (!!!), they work for the first iteration of the design. We must evaluate
which ones we need to optimize (read papers, run parametric sweeps etc) and which ones we just need a realistic value
"""
# Board
L_BOARD = 153.0             # TODO
W_BOARD = 120.0             # TODO

# Slot side
D_CAV = 19.0                # circular cavity diameter TODO
X_CAV = 10.0                # back edge of board -> back of the cavity TODO
L_SLOT = 4.0                # uniform slotline length between cavity and flare throat TODO
W_SLOT = 0.25               # slotline width, needs to be low for impedance purposes TODO
W_APERTURE = 80.0           # flare opening at x = L_BOARD (lambda_0/2 at F_LO = 75 mm, plus margin) TODO
R_TAPER = 0.0481            # opening rate R, 1/mm = ln(W_APERTURE / W_SLOT) / 120 -> exponential opening TODO
N_TAPER = 120               # number of points used to draw each flare edge TODO

# Feed side
W_MS = 1.53                 # 50-ohm microstrip width on 0.8 mm FR4 TODO
W_MS_X = 0.35               # ~100-ohm width where the strip crosses the slot -> read slotline paper!!! TODO
L_MS_TAPER = 25.0           # length of the linear 50 -> 100 ohm width taper TODO
L_MS_NARROW = 3.0           # narrow (W_MS_X) section before the slot centre, after the taper TODO
FEED_OFFSET = 2.0           # cavity edge -> microstrip centre line (along the slotline) TODO
MS_OVERSHOOT = 1.0          # how far the strip continues past the slot centre before the stub TODO
R_STUB = 8.5                # radial stub radius (~ quarter guided wavelength near 5 GHz; sets high end) TODO
A_STUB = 90.0               # radial stub angle, degrees TODO
STUB_TILT = 25.0            # stub axis rotated from +y towards +x, degrees (keeps the stub off the cavity) TODO
STUB_CLEARANCE = 1.0        # minimum allowed gap between the stub and the cavity TODO
N_STUB = 16                 # segments used for the stub arc TODO

# Optional slot port (--slot-port), same as the cut model of feed_compare_hfss.py
SLOT_PORT_DIST = 15.0       # microstrip centre line -> slot port (= L_OUT in feed_compare_hfss.py)

# Solve settings, for when we use --solve
USE_COPPER = True           # True: finite-conductivity copper (conductor loss); False: PEC
MAX_PASSES = 15             # adaptive mesh passes, technically not necessary
MAX_DELTA_S = 0.02          # adaptive convergence target, this is the standard target. For more precise simulations
                            # (i.e., for finetuning and final graphs) set MAX_DELTA_S = 0.01 or 0.005
SWEEP_POINTS = 701          # points in the interpolating S-parameter sweep
FIELD_STEP = 0.5            # GHz, spacing of the discrete sweep that saves fields (F_LO ... F_HI)
PLOT_FREQS = [2.0, 4.5, 7.0]  # GHz, frequencies for patterns and field overlays (they lie on the FIELD_STEP grid!!!)
FF_STEP = 2                 # degrees, angular step of the far-field spheres, only for nice graphics

# Derived positions
X_CAV_C = X_CAV + D_CAV / 2.0           # cavity centre
X_THROAT = X_CAV + D_CAV + L_SLOT       # where the flare starts
X_FEED = X_CAV + D_CAV + FEED_OFFSET    # microstrip centre line


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
def taper_half_width(x):
    # Half-width of the flare at position x (exponential Vivaldi :)
    x1, y1 = X_THROAT, W_SLOT / 2.0
    x2, y2 = L_BOARD, W_APERTURE / 2.0
    e1, e2 = math.exp(R_TAPER * x1), math.exp(R_TAPER * x2)
    c1 = (y2 - y1) / (e2 - e1)
    c2 = (y1 * e2 - y2 * e1) / (e2 - e1)
    return c1 * math.exp(R_TAPER * x) + c2


def flare_outline():
    # Polygon (list of [x, y]) of the flared part of the slot (see Convex Opt)
    xs = [X_THROAT + (L_BOARD - X_THROAT) * i / (N_TAPER - 1) for i in range(N_TAPER)]
    upper = [[x, taper_half_width(x)] for x in xs]
    lower = [[x, -taper_half_width(x)] for x in reversed(xs)]
    return upper + lower


def feed_outline():
    # "Polygon" of the microstrip: 50-ohm from the board edge, linear taper, narrow across the slot
    y_edge = -W_BOARD / 2.0
    y_t1 = -L_MS_NARROW                 # taper ends, narrow section starts
    y_t0 = y_t1 - L_MS_TAPER            # taper starts
    a, b = W_MS / 2.0, W_MS_X / 2.0
    return [[X_FEED - a, y_edge], [X_FEED + a, y_edge], [X_FEED + a, y_t0], [X_FEED + b, y_t1],
            [X_FEED + b, MS_OVERSHOOT], [X_FEED - b, MS_OVERSHOOT], [X_FEED - b, y_t1], [X_FEED - a, y_t0]]


def stub_outline():
    # Closed polygon of the radial stub, fanning towards +y and tilted by STUB_TILT towards +x.
    # The apex sits slightly inside the strip end so the stub and the strip overlap over the full
    # strip width (an apex exactly on the strip end would only touch it at a single point).
    half = math.radians(A_STUB / 2.0)
    tilt = math.radians(STUB_TILT)
    apex = [X_FEED, MS_OVERSHOOT - (W_MS_X / 2.0) / math.tan(half)]
    pts = [apex]
    for i in range(N_STUB + 1):
        a = tilt - half + 2.0 * half * i / N_STUB    # angle measured from the +y axis
        pts.append([apex[0] + R_STUB * math.sin(a), apex[1] + R_STUB * math.cos(a)])
    return pts


def stub_clearance():
    # Smallest gaps between the stub (beyond the strip end) and the slot openings.
    # Returns (gap to the cavity, gap to the slotline / flare).
    apex, arc = stub_outline()[0], stub_outline()[1:]
    gap_cav = gap_slot = float("inf")
    for px, py in arc:
        for k in range(1, 41):
            x = apex[0] + (px - apex[0]) * k / 40.0
            y = apex[1] + (py - apex[1]) * k / 40.0
            if y <= MS_OVERSHOOT:
                continue                                              # still inside the strip
            gap_cav = min(gap_cav, math.hypot(x - X_CAV_C, y) - D_CAV / 2.0)
            if x >= X_CAV_C:
                slot_half = taper_half_width(x) if x > X_THROAT else W_SLOT / 2.0
                gap_slot = min(gap_slot, abs(y) - slot_half)
    return gap_cav, gap_slot


def slot_impedance():
    # Slotline impedance Zos of W_SLOT from code_python/calcul.py (same value as the cut-model slot port)
    import contextlib, io, runpy
    with contextlib.redirect_stdout(io.StringIO()):                    # calcul.py prints its table
        c = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_python", "calcul.py"))
        return round(c["calculZos"](EPS_R, W_SLOT * 1e-3, c["lamda0s"], T_SUB * 1e-3), 2)


def slot_port_sheet():
    # (x0, half-width) of the slot-port sheet: it spans x0 .. x0 + W_SLOT and the local slot opening
    x0 = X_FEED + SLOT_PORT_DIST - W_SLOT
    xm = x0 + W_SLOT / 2.0
    return x0, (taper_half_width(xm) if xm > X_THROAT else W_SLOT / 2.0)


def check_parameters():
    # Here we check
    assert X_FEED + SLOT_PORT_DIST < L_BOARD, "Slot port lies outside the board."
    assert X_THROAT < L_BOARD, "Cavity + slotline are longer than the board."
    assert W_APERTURE < W_BOARD, "Aperture must be narrower than the board."
    assert W_SLOT < D_CAV, "Slotline must be narrower than the cavity."
    assert X_CAV + D_CAV < X_FEED - W_MS_X / 2 and X_FEED + W_MS_X / 2 < X_THROAT, \
        "Microstrip must cross the uniform slotline section (check FEED_OFFSET / L_SLOT)."
    assert L_MS_NARROW + L_MS_TAPER < W_BOARD / 2, "Microstrip taper does not fit between the edge and the slot."
    assert math.hypot(X_FEED + W_MS / 2 - X_CAV_C, L_MS_NARROW) > D_CAV / 2 or X_FEED - W_MS / 2 > X_CAV + D_CAV, \
        "The wide part of the microstrip runs over the cavity."
    gap_cav, gap_slot = stub_clearance()
    assert gap_cav >= STUB_CLEARANCE, \
        f"Radial stub is {gap_cav:.2f} mm from the cavity (< {STUB_CLEARANCE} mm): change STUB_TILT / A_STUB / R_STUB."
    assert gap_slot >= 0.5, \
        f"Radial stub is {gap_slot:.2f} mm from the slotline / flare: reduce STUB_TILT or raise MS_OVERSHOOT."
    grid = [round(F_LO + i * FIELD_STEP, 6) for i in range(int(round((F_HI - F_LO) / FIELD_STEP)) + 1)]
    assert all(round(f, 6) in grid for f in PLOT_FREQS), "PLOT_FREQS must lie on the FIELD_STEP grid."


# --------------------------------------------------------------------------------------
# Matplotlib preview
# --------------------------------------------------------------------------------------
def preview(path="vivaldi_preview.png", slot_port=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.add_patch(Rectangle((0, -W_BOARD / 2), L_BOARD, W_BOARD, fc="#c9a25a", ec="k", label="ground metal (top)"))
    slot = dict(fc="white", ec="k")
    ax.add_patch(Circle((X_CAV_C, 0), D_CAV / 2, **slot))
    ax.add_patch(Rectangle((X_CAV_C, -W_SLOT / 2), X_THROAT - X_CAV_C, W_SLOT, fc="white", ec="none"))
    ax.add_patch(Polygon(flare_outline(), **slot))
    feed = dict(fc="#1f77b4", ec="none", alpha=0.6)
    ax.add_patch(Polygon(feed_outline(), label="microstrip + stub (bottom)", **feed))
    ax.add_patch(Polygon(stub_outline(), **feed))
    ax.plot([X_FEED], [-W_BOARD / 2], "rv", ms=10, label="lumped port")
    if slot_port:
        ax.plot([X_FEED + SLOT_PORT_DIST], [0], "g<", ms=10, label=f"slot port ({slot_impedance()} ohm)")
    ax.set_xlim(-5, L_BOARD + 5)
    ax.set_ylim(-W_BOARD / 2 - 5, W_BOARD / 2 + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]  (radiation ->)")
    ax.set_ylabel("y [mm]")
    ax.legend(loc="upper left")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"Preview written to {path}")


# --------------------------------------------------------------------------------------
# HFSS model
# --------------------------------------------------------------------------------------
def build(solve=False, non_graphical=False, version=None, gif=False, slot_port=False):
    from ansys.aedt.core import Hfss          # older releases: from pyaedt import Hfss

    design = "Vivaldi_microstrip_fed_P2" if slot_port else "Vivaldi_microstrip_fed"
    hfss = Hfss(project="Vivaldi", design=design, solution_type="Modal",
                version=version, non_graphical=non_graphical, new_desktop=True)
    hfss.modeler.model_units = "mm"
    m = hfss.modeler

    def polygon(xy, z, name):
        """Covered sheet from a list of [x, y] points at height z."""
        return m.create_polyline([[x, y, z] for x, y in xy], cover_surface=True,
                                 close_surface=True, name=name)

    # ---- substrate -------------------------------------------------------------------
    mat = hfss.materials.add_material("Substrate_FR4")
    mat.permittivity = EPS_R
    mat.dielectric_loss_tangent = TAN_D
    m.create_box([0, -W_BOARD / 2, 0], [L_BOARD, W_BOARD, T_SUB], name="Substrate",
                 material="Substrate_FR4")

    # ---- top metal: ground plane minus (cavity + slotline + flare) -------------------
    ground = m.create_rectangle("XY", [0, -W_BOARD / 2, T_SUB], [L_BOARD, W_BOARD], name="Ground")
    cavity = m.create_circle("XY", [X_CAV_C, 0, T_SUB], D_CAV / 2, name="Cavity")
    # the slotline starts at the cavity centre so the two shapes overlap cleanly
    slotline = m.create_rectangle("XY", [X_CAV_C, -W_SLOT / 2, T_SUB],
                                  [X_THROAT - X_CAV_C, W_SLOT], name="Slotline")
    flare = polygon(flare_outline(), T_SUB, "Flare")
    m.unite([cavity, slotline, flare])
    m.subtract(ground, cavity, keep_originals=False)

    # ---- bottom metal: tapered microstrip + radial stub ------------------------------
    strip = polygon(feed_outline(), 0, "Feed")
    stub = polygon(stub_outline(), 0, "Stub")
    m.unite([strip, stub])

    for obj, colour in ((ground, (200, 160, 60)), (strip, (30, 110, 180))):
        obj.color = colour
    if USE_COPPER:
        hfss.assign_finite_conductivity([ground.name, strip.name], material="copper", name="Metal")
    else:
        hfss.assign_perfecte_to_sheets([ground.name, strip.name], name="Metal")

    # ---- lumped port at the board edge, strip (z = 0) to ground (z = h) ---------------
    y_edge = -W_BOARD / 2
    x0, x1 = X_FEED - W_MS / 2, X_FEED + W_MS / 2
    port = m.create_polyline([[x0, y_edge, 0], [x1, y_edge, 0], [x1, y_edge, T_SUB], [x0, y_edge, T_SUB]],
                             cover_surface=True, close_surface=True, name="PortSheet")
    hfss.lumped_port(assignment=port.name, create_port_sheet=False,
                     integration_line=[[X_FEED, y_edge, 0], [X_FEED, y_edge, T_SUB]],
                     impedance=50, name="P1", renormalize=True)

    # ---- optional slot port: across the slot opening, SLOT_PORT_DIST past the strip, Zos
    if slot_port:
        xs, hw = slot_port_sheet()
        sheet = m.create_rectangle("XY", [xs, -hw, T_SUB], [W_SLOT, 2 * hw], name="PortSheet2")
        xm = xs + W_SLOT / 2.0
        hfss.lumped_port(assignment=sheet.name, create_port_sheet=False,
                         integration_line=[[xm, -hw, T_SUB], [xm, hw, T_SUB]],
                         impedance=slot_impedance(), name="P2", renormalize=True)

    # ---- radiation boundary (air box sized for the lowest frequency) ------------------
    hfss.create_open_region(frequency=f"{F_LO}GHz", boundary="Radiation")

    # ---- analysis ---------------------------------------------------------------------
    # Broadband adaptive meshing refines the mesh for the whole F_LO-F_HI band instead of F_HI only.
    setup = hfss.create_setup("Setup1", Frequency=f"{F_HI}GHz", MaximumPasses=MAX_PASSES,
                              MaxDeltaS=MAX_DELTA_S, MinimumConvergedPasses=2)
    try:
        ok = setup.enable_adaptive_setup_broadband(f"{F_LO}GHz", f"{F_HI}GHz", MAX_PASSES, MAX_DELTA_S)
    except Exception as err:                                   # noqa: BLE001
        ok = False
        print(f"[warning] broadband adaptive not set: {err}")
    if not ok:
        print(f"[warning] falling back to single-frequency adaptive meshing at {F_HI} GHz")

    # Fast, dense S-parameter sweep (no fields)
    hfss.create_linear_count_sweep(setup.name, "GHz", F_LO - 1.0, F_HI + 1.0, num_of_freq_points=SWEEP_POINTS,
                                   name="Sweep", sweep_type="Interpolating", save_fields=False)
    # Discrete sweep with saved fields: needed for far fields and field overlays at several frequencies
    n_field = int(round((F_HI - F_LO) / FIELD_STEP)) + 1
    hfss.create_linear_count_sweep(setup.name, "GHz", F_LO, F_HI, num_of_freq_points=n_field,
                                   name="Fields", sweep_type="Discrete", save_fields=True)

    # Far-field spheres: a full sphere for 3D patterns and the E-plane, and a full XZ circle for the H-plane
    hfss.insert_infinite_sphere(definition="Theta-Phi", theta_start=0, theta_stop=180, theta_step=FF_STEP,
                                phi_start=0, phi_stop=360, phi_step=FF_STEP, name="Sphere3D")
    hfss.insert_infinite_sphere(definition="Theta-Phi", theta_start=-180, theta_stop=180, theta_step=1,
                                phi_start=0, phi_stop=90, phi_step=90, name="HplaneCut")

    m.fit_all()
    hfss.save_project()

    if solve:
        hfss.analyze(setup=setup.name)
        hfss.save_project()
        postprocess(hfss, setup.name, ground.name, strip.name, gif=gif, slot_port=slot_port)
        hfss.save_project()

    print(f"Project saved to {hfss.project_file}")
    return hfss


# --------------------------------------------------------------------------------------
# Post-processing: reports, field overlays, exports
# --------------------------------------------------------------------------------------
def postprocess(hfss, setup_name, ground_name, strip_name, gif=False, slot_port=False):
    post = hfss.post
    m = hfss.modeler
    s_sweep = f"{setup_name} : Sweep"       # interpolating sweep, S-parameters only
    f_sweep = f"{setup_name} : Fields"      # discrete sweep with saved fields
    out_dir = os.path.dirname(hfss.project_file)
    done, failed = [], []

    def attempt(label, func, *args, **kwargs):
        """Run one post-processing step; a failure is reported but does not stop the others."""
        try:
            result = func(*args, **kwargs)
            if result is False:
                raise RuntimeError("AEDT returned False")
            done.append(label)
            return result
        except Exception as err:                               # noqa: BLE001
            failed.append(f"{label}: {err}")
            return None

    # Name of the port as it appears in the S-parameter expressions (normally "P1")
    try:
        port = hfss.get_traces_for_plot(category="S")[0][2:-1].split(",")[0]
    except Exception:                                          # noqa: BLE001
        port = "P1"

    # ---- circuit quantities (dense interpolating sweep) -------------------------------
    attempt("S11", post.create_report, f"dB(S({port},{port}))", s_sweep, plot_name="S11")
    attempt("VSWR", post.create_report, f"VSWR({port})", s_sweep, plot_name="VSWR")
    attempt("Zin", post.create_report, [f"re(Z({port},{port}))", f"im(Z({port},{port}))"], s_sweep,
            plot_name="Input impedance")
    attempt("Smith", post.create_report, f"S({port},{port})", s_sweep, plot_type="Smith Chart",
            plot_name="Smith chart")
    if slot_port:
        # same quantities as the cut model of feed_compare_hfss.py
        attempt("S21", post.create_report, [f"dB(S({port},{port}))", f"dB(S(P2,{port}))"], s_sweep,
                plot_name="S-parameters")
        attempt("Power balance", post.create_report, [f"1-mag(S({port},{port}))^2-mag(S(P2,{port}))^2"], s_sweep,
                plot_name="Radiated + lost power fraction")
        # far fields with the antenna driven from the microstrip only (P2 stays a matched load)
        attempt("Sources", hfss.edit_sources, {f"{port}:1": ("1W", "0deg"), "P2:1": ("0W", "0deg")})
    attempt("Touchstone", hfss.export_touchstone, setup_name, "Sweep",
            os.path.join(out_dir, "vivaldi_P2.s2p" if slot_port else "vivaldi.s1p"))

    # ---- radiation versus frequency (discrete sweep) ----------------------------------
    # Endfire direction is +x: Theta = 90 deg, Phi = 0 deg
    boresight = {"Freq": ["All"], "Theta": ["90deg"], "Phi": ["0deg"]}
    attempt("Endfire gain vs f", post.create_report,
            ["dB(RealizedGainTotal)", "dB(GainTotal)", "dB(DirTotal)"], f_sweep, variations=boresight,
            primary_sweep_variable="Freq", report_category="Far Fields", context="Sphere3D",
            plot_name="Endfire gain vs frequency")
    attempt("Peak gain vs f", post.create_report,
            ["dB(PeakRealizedGain)", "dB(PeakGain)", "dB(PeakDirectivity)"], f_sweep,
            variations={"Freq": ["All"]}, primary_sweep_variable="Freq",
            report_category="Antenna Parameters", context="Sphere3D", plot_name="Peak gain vs frequency")
    attempt("Efficiency vs f", post.create_report, ["RadiationEfficiency"], f_sweep,
            variations={"Freq": ["All"]}, primary_sweep_variable="Freq",
            report_category="Antenna Parameters", context="Sphere3D", plot_name="Radiation efficiency")
    attempt("Front-to-back vs f", post.create_report, ["dB(FrontToBackRatio)"], f_sweep,
            variations={"Freq": ["All"]}, primary_sweep_variable="Freq",
            report_category="Antenna Parameters", context="Sphere3D", plot_name="Front-to-back ratio")

    # ---- pattern cuts with one trace per plot frequency -------------------------------
    freqs = [f"{f}GHz" for f in PLOT_FREQS]
    attempt("E-plane cut", post.create_report, "dB(RealizedGainTotal)", f_sweep,
            variations={"Freq": freqs, "Theta": ["90deg"], "Phi": ["All"]}, primary_sweep_variable="Phi",
            report_category="Far Fields", plot_type="Radiation Pattern", context="Sphere3D",
            plot_name="E-plane (XY) realized gain")
    attempt("H-plane cut", post.create_report, "dB(RealizedGainTotal)", f_sweep,
            variations={"Freq": freqs, "Phi": ["0deg"], "Theta": ["All"]}, primary_sweep_variable="Theta",
            report_category="Far Fields", plot_type="Radiation Pattern", context="HplaneCut",
            plot_name="H-plane (XZ) realized gain")
    attempt("Cross-pol E-plane", post.create_report,
            ["dB(RealizedGainPhi)", "dB(RealizedGainTheta)"], f_sweep,
            variations={"Freq": [freqs[len(freqs) // 2]], "Theta": ["90deg"], "Phi": ["All"]},
            primary_sweep_variable="Phi", report_category="Far Fields", context="Sphere3D",
            plot_name="E-plane co-pol (Phi) and cross-pol (Theta)")

    # ---- 3D patterns and field overlays at each plot frequency ------------------------
    # Make the air region and the substrate see-through, otherwise they hide the overlays
    obj_names = set(m.object_names)
    for name, trans in (("Region", 1.0), ("CreateRegion", 1.0), ("Substrate", 0.7)):
        if name in obj_names:
            attempt(f"Transparency {name}", setattr, m[name], "transparency", trans)

    # Cut plane through the middle of the substrate (cleaner than z = 0, which holds the microstrip)
    xy_plane = "Global:XY"
    if attempt("Mid-substrate plane", m.create_coordinate_system, origin=[0, 0, T_SUB / 2.0], name="MidSub"):
        xy_plane = "MidSub:XY"
    attempt("Reset working CS", m.set_working_coordinate_system, "Global")

    for f in PLOT_FREQS:
        tag = f"{f:g}GHz".replace(".", "p")
        freq = f"{f}GHz"
        attempt(f"3D pattern {freq}", post.create_report, "dB(RealizedGainTotal)", f_sweep,
                variations={"Freq": [freq], "Theta": ["All"], "Phi": ["All"]},
                primary_sweep_variable="Phi", secondary_sweep_variable="Theta",
                report_category="Far Fields", plot_type="3D Polar Plot", context="Sphere3D",
                plot_name=f"3D realized gain {tag}")
        intr = {"Freq": freq, "Phase": "0deg"}
        attempt(f"E-field XY {freq}", post.create_fieldplot_cutplane, [xy_plane], "Mag_E", setup=f_sweep,
                intrinsics=intr, plot_name=f"E_XY_{tag}")
        attempt(f"E-field XZ {freq}", post.create_fieldplot_cutplane, ["Global:XZ"], "Mag_E", setup=f_sweep,
                intrinsics=intr, plot_name=f"E_XZ_{tag}")
        attempt(f"Surface current {freq}", post.create_fieldplot_surface, [ground_name, strip_name], "Mag_Jsurf",
                setup=f_sweep, intrinsics=intr, plot_name=f"Jsurf_{tag}")

        if gif:
            phases = [f"{p}deg" for p in range(0, 360, 20)]
            attempt(f"GIF {freq}", post.plot_animated_field, "Mag_E", [xy_plane], plot_type="CutPlane",
                    setup=f_sweep, intrinsics={"Freq": freq}, variation_variable="Phase", variations=phases,
                    view="xy", show=False, export_gif=True, export_path=out_dir)

    print(f"\nPost-processing: {len(done)} steps OK, {len(failed)} failed")
    for line in failed:
        print(f"  [failed] {line}")
    print(
        "\nTo display the overlays in the AEDT GUI:\n"
        "  1. Project tree -> Field Overlays -> tick the checkbox of one plot (e.g. E_XY_4p5GHz)\n"
        "  2. Right-click Field Overlays -> Plot Fields -> Delete/Untick the others (one at a time reads best)\n"
        "  3. If the model still hides the plot: right-click the plot -> Apply Ghosted (translucent model)\n"
        "  4. To overlay the 3D far-field pattern on the model: right-click Field Overlays ->\n"
        "     Plot Fields -> Radiation Field, then pick one of the '3D realized gain' reports\n"
        "  5. To watch the wave radiate: right-click an E_XY / E_XZ plot -> Animate (sweep Phase 0-360deg)\n"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true", help="draw the outline only, without HFSS")
    ap.add_argument("--solve", action="store_true", help="run the simulation after building")
    ap.add_argument("--gif", action="store_true", help="with --solve: also export animated E-field GIFs (pyvista)")
    ap.add_argument("--slot-port", action="store_true",
                    help="add the slot port of the feed_compare_hfss.py cut model (design ..._P2)")
    ap.add_argument("--non-graphical", action="store_true", help="run HFSS without the GUI")
    ap.add_argument("--version", default=None, help='AEDT version, e.g. "2025.1" (default: newest installed)')
    args = ap.parse_args()

    check_parameters()
    if args.preview:
        preview(slot_port=args.slot_port)
    else:
        app = build(solve=args.solve, non_graphical=args.non_graphical, version=args.version, gif=args.gif,
                    slot_port=args.slot_port)
        if args.non_graphical:
            app.release_desktop()
