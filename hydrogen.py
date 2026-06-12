import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import py3Dmol
from pyscf import gto, scf, mp, mcscf

HARTREE2EV = 27.211386


# ----------------------------------------------------------------------------
# Geometry: two H atoms on the z axis, separation = bond (Angstrom)
# ----------------------------------------------------------------------------
def build_mol(bond, basis):
    return gto.M(atom=[["H", (0, 0, 0)], ["H", (bond, 0, 0)]],
                 basis=basis, unit="Angstrom", verbose=0)


# ----------------------------------------------------------------------------
# Robust UHF: compare the symmetric solution against a broken-symmetry guess
# (HOMO/LUMO mixed oppositely for alpha and beta) and keep the lower one.
# ----------------------------------------------------------------------------
def run_uhf(mol):
    mf_sym = scf.UHF(mol).run()
    rhf = scf.RHF(mol).run()
    mo, nocc = rhf.mo_coeff, mol.nelectron // 2
    homo, lumo, t = nocc - 1, nocc, np.pi / 4
    ca, cb = mo.copy(), mo.copy()
    ca[:, homo] = np.cos(t) * mo[:, homo] + np.sin(t) * mo[:, lumo]
    cb[:, homo] = np.cos(t) * mo[:, homo] - np.sin(t) * mo[:, lumo]
    dma = ca[:, :nocc] @ ca[:, :nocc].T
    dmb = cb[:, :nocc] @ cb[:, :nocc].T
    mf_brk = scf.UHF(mol)
    mf_brk.kernel(dm0=np.array([dma, dmb]))
    return mf_brk if mf_brk.e_tot < mf_sym.e_tot - 1e-9 else mf_sym


# ----------------------------------------------------------------------------
# Cached solvers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def solve(bond, basis):
    mol = build_mol(bond, basis)
    rhf = scf.RHF(mol).run()
    uhf = run_uhf(mol)
    mp2 = mp.MP2(rhf).run()
    mc = mcscf.CASSCF(rhf, 2, 2).run()
    occ = np.sort(np.linalg.eigvalsh(mc.fcisolver.make_rdm1(mc.ci, 2, 2)))[::-1]
    return dict(e_rhf=rhf.e_tot, e_uhf=uhf.e_tot, e_mp2=mp2.e_tot, e_cas=mc.e_tot,
                ss=uhf.spin_square()[0], mp2_corr=mp2.e_corr, cas_occ=occ,
                nmo=mol.nao, nocc=mol.nelectron // 2)


@st.cache_data(show_spinner=False)
def curve(basis, npts=40):
    rs = np.linspace(0.40, 4.00, npts)
    keys = ["e_rhf", "e_uhf", "e_mp2", "e_cas"]
    E = {k: [] for k in keys}
    occ = []
    for r in rs:
        s = solve(float(r), basis)
        for k in keys:
            E[k].append(s[k])
        occ.append(s["cas_occ"])
    return rs, {k: np.array(v) for k, v in E.items()}, np.array(occ)


@st.cache_data(show_spinner=False)
def orbital_cube(bond, basis, method, spin, idx, ngrid):
    from pyscf.tools import cubegen
    mol = build_mol(bond, basis)
    if method == "RHF":
        coeff = scf.RHF(mol).run().mo_coeff[:, idx]
    elif method == "UHF":
        uhf = run_uhf(mol)
        coeff = uhf.mo_coeff[0 if spin == "alpha" else 1][:, idx]
    else:  # CASSCF natural orbitals
        mc = mcscf.CASSCF(scf.RHF(mol).run(), 2, 2).run()
        _, natorbs = mcscf.addons.make_natural_orbitals(mc)
        coeff = natorbs[:, idx]
    tmp = tempfile.NamedTemporaryFile(suffix=".cube", delete=False).name
    cubegen.orbital(mol, tmp, coeff, nx=ngrid, ny=ngrid, nz=ngrid)
    with open(tmp) as fh:
        return fh.read()


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="H2 dissociation explorer", layout="wide")
st.title("H₂ dissociation explorer")
st.caption("RHF · UHF · MP2 · CASSCF(2,2) — pull the bond apart and watch each method succeed or fail.")

with st.sidebar:
    st.header("System")
    bond = st.slider("H–H bond length (Å)", 0.40, 4.00, 0.74, 0.02)
    basis = st.selectbox("Basis set", ["sto-3g", "6-31g", "cc-pvdz"], index=0)
    st.header("Orbital viewer")
    method = st.radio("Orbitals from", ["UHF", "RHF", "CASSCF nat. orb."], index=0)
    spin = st.radio("Spin channel", ["alpha", "beta"], horizontal=True,
                    disabled=(method != "UHF"))
    iso = st.slider("Isosurface value", 0.01, 0.10, 0.04, 0.005)
    ngrid = st.select_slider("Grid resolution", [30, 40, 50, 60], value=40)

s = solve(bond, basis)
nmo, nocc = s["nmo"], s["nocc"]
idx = st.sidebar.selectbox("Orbital index", range(nmo), index=nocc - 1,
                           format_func=lambda i: f"#{i} ({'occ' if i < nocc else 'virt'})")

# --- energies ---
st.subheader("Energies at this bond length")
c1, c2, c3, c4 = st.columns(4)
c1.metric("RHF", f"{s['e_rhf']:.5f} Ha")
c2.metric("UHF", f"{s['e_uhf']:.5f} Ha", f"{(s['e_uhf']-s['e_rhf'])*1000:.1f} mHa vs RHF")
c3.metric("MP2", f"{s['e_mp2']:.5f} Ha")
c4.metric("CASSCF(2,2)", f"{s['e_cas']:.5f} Ha")

# --- diagnostics ---
d1, d2, d3, d4 = st.columns(4)
d1.metric("UHF ⟨S²⟩", f"{s['ss']:.3f}", help="0 = pure singlet (restricted); ~1 = fully spin-broken at dissociation")
d2.metric("MP2 correlation", f"{s['mp2_corr']*1000:.1f} mHa")
d3.metric("CAS occ σ", f"{s['cas_occ'][0]:.3f}", help="2.0 = clean bond; -> 1.0 = static correlation")
d4.metric("CAS occ σ*", f"{s['cas_occ'][1]:.3f}", help="0.0 = clean bond; -> 1.0 = static correlation")

left, right = st.columns([3, 2])

# --- dissociation curves (the hero plot) ---
with left:
    rs, E, occ = curve(basis)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(rs, E["e_rhf"], color="tab:blue",   label="RHF")
    ax.plot(rs, E["e_uhf"], color="tab:green",  label="UHF")
    ax.plot(rs, E["e_mp2"], color="tab:orange", label="MP2")
    ax.plot(rs, E["e_cas"], color="tab:red",    label="CASSCF(2,2)", lw=2.4)
    ax.axvline(bond, ls="--", lw=1, color="0.5")
    ax.plot(bond, s["e_cas"], "o", color="tab:red", ms=7)
    lo = E["e_cas"].min() - 0.06
    hi = max(E["e_rhf"].max(), E["e_uhf"].max()) + 0.05
    ax.set_ylim(lo, hi)                       # lets MP2 dive off-scale where it fails
    ax.set_xlabel("H–H bond length (Å)")
    ax.set_ylabel("total energy (Ha)")
    ax.set_title("Potential energy curves")
    ax.legend(fontsize=9)
    st.pyplot(fig)
    plt.close(fig)

# --- 3D orbital viewer ---
with right:
    m_key = "CASSCF" if method.startswith("CASSCF") else method
    cube = orbital_cube(bond, basis, m_key, spin, idx, ngrid)
    view = py3Dmol.view(width=420, height=380)
    view.addModel(cube, "cube")
    view.setStyle({"stick": {"radius": 0.10}, "sphere": {"scale": 0.20}})
    view.addVolumetricData(cube, "cube", {"isoval": iso,  "color": "red",  "opacity": 0.85})
    view.addVolumetricData(cube, "cube", {"isoval": -iso, "color": "blue", "opacity": 0.85})
    view.zoomTo()
    label = f"{method}" + (f" {spin}" if method == "UHF" else "") + f"  ·  orbital #{idx}"
    st.markdown(f"**{label}**")
    components.html(view._make_html(), height=400)
    st.caption("Red / blue = sign of the wavefunction. Drag to rotate.")

# --- CAS natural occupations vs bond ---
with st.expander("CASSCF natural occupations vs bond length  (static correlation turning on)", expanded=True):
    fig2, ax2 = plt.subplots(figsize=(7, 2.6))
    ax2.plot(rs, occ[:, 0], color="tab:red",  label="σ  (bonding)")
    ax2.plot(rs, occ[:, 1], color="tab:blue", label="σ* (antibonding)")
    ax2.axhline(1.0, ls=":", lw=1, color="0.6")
    ax2.axvline(bond, ls="--", lw=1, color="0.5")
    ax2.set_xlabel("H–H bond length (Å)")
    ax2.set_ylabel("occupation")
    ax2.set_ylim(-0.05, 2.05)
    ax2.legend(fontsize=8)
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption("Near equilibrium the bond is a clean σ² (2 / 0). As it stretches both "
               "orbitals head toward 1.0 — the wavefunction becomes genuinely two-configurational, "
               "which is exactly the static correlation a single determinant cannot capture.")

with st.expander("Things to try"):
    st.markdown(
        "- **Find the Coulson–Fischer point.** Stretch past ~1.2 Å and watch the green UHF "
        "curve peel *below* the blue RHF curve while ⟨S²⟩ climbs from 0 toward 1.\n"
        "- **See symmetry breaking in 3D.** With the viewer on UHF, stretch the bond and flip "
        "between α and β — the two spins localize on *opposite* atoms. That spatial separation "
        "is the broken symmetry.\n"
        "- **Watch RHF fail.** At long bond length RHF sits far too high (it insists on an equal "
        "ionic/covalent mix), while CASSCF dissociates correctly to two H atoms.\n"
        "- **Watch MP2 fail.** Switch the basis to cc-pVDZ and stretch — MP2 plunges off the "
        "bottom of the plot as its perturbation denominators collapse.\n"
        "- **Watch correlation turn on.** Follow the occupation plot from 2/0 to 1/1."
    )
