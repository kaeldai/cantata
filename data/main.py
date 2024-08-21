#!/usr/bin/env python3

from collections import defaultdict
import arbor as A
from arbor import units as U
import pandas as pd
from json import load as load_data
import matplotlib.pyplot as plt
from time import perf_counter as pc
import re
from pathlib import Path

here = Path(__file__).parent

have_timing = True
have_stats = True

# check arbor version
ver = re.match(r"(\d+)\.(\d+)\.(\d+)(-\w+)?", A.__version__)
if ver:
    mj, mn, pt, sf = ver.groups()
    assert (
        10000 <= (int(mj) * 1000 + int(mn)) * 1000 + int(pt) <= 11000
    ), "Arbor version 0.10.x is required."
else:
    print(f"Couldn't parse version {A.__version__}")
    exit(-42)


def load_morphology(path):
    sfx = path.suffix
    if sfx == ".swc":
        try:
            return A.load_swc_arbor(path).morphology
        except:
            pass
        try:
            return A.load_swc_neuron(path).morphology
        except:
            raise RuntimeError(
                f"Could load {path} neither as NEURON nor Arbor flavour."
            )
    elif sfx == ".asc":
        return A.load_asc(path).morphology
    elif sfx == ".nml":
        nml = A.neuroml(path)
        if len(nml.morphology_ids()) == 1:
            return nml.morphology(nml.morphology_ids()[0])
        else:
            raise RuntimeError(f"NML file {path} contains multiple morphologies.")
    else:
        raise RuntimeError(f"Unknown morphology file type {sfx}")


if have_timing:

    class Timing:
        def __init__(self):
            self.timings = defaultdict(lambda: 0.0)

        def tic(self, key):
            self.timings[key] -= pc()

        def toc(self, key):
            self.timings[key] += pc()

else:

    class Timing:
        def __init__(self):
            pass

        def tic(self, _):
            pass

        def toc(self, _):
            pass


timing = Timing()


def open_sim():
    with open(here / "dat/sim.json", "rb") as fd:
        return load_data(fd)


def close_sim(raw):
    del raw


def read_int_dict(raw, key):
    return {int(k): v for k, v in raw[key].items()}


def read_dict(raw, key):
    res = raw[key]
    assert isinstance(res, dict)
    return res


def read_array(raw, key):
    res = raw[key]
    assert isinstance(res, list)
    return res


def read_int(raw, key):
    res = raw[key]
    assert isinstance(res, int)
    return res


def read_float(raw, key):
    res = raw[key]
    assert isinstance(res, float)
    return res


class recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)

        data = open_sim()

        # Some initial, global data
        self.N = read_int(data, "size")
        self.T = read_float(data, "time")
        self.dt = read_float(data, "time_step")
        self.threshold = read_float(data, "spike_threshold")

        # gid -> (morphology id, acc id)
        self.gid_to_bio = read_int_dict(data, "cell_bio_ids")
        # gid -> lif cell data
        self.gid_to_lif = read_int_dict(data, "gid_to_lif")
        # gid -> virtual cell data
        self.gid_to_vrt = read_int_dict(data, "gid_to_vrt")
        # morphology id -> morphology resource file
        self.mid_to_mrf = read_array(data, "morphology")
        # cell id -> decor file
        self.cid_to_acc = read_array(data, "decoration")
        # gid -> cell kind
        self.gid_to_kid = read_array(data, "cell_kind")
        # gid -> cell metadata
        self.gid_to_meta = read_array(data, "metadata")
        # gid -> incoming connections
        self.gid_to_inc = read_int_dict(data, "incoming_connections")
        # gid -> synapse
        self.gid_to_syn = read_int_dict(data, "synapses")
        # gid -> iclamps. NOTE must only be set if kind[gid] == cable
        self.gid_to_icp = read_int_dict(data, "current_clamps")
        # probes
        self.gid_to_prb = read_int_dict(data, "probes")
        # cell kind specific counts
        self.kind_to_count = read_array(data, "count_by_kind")

        # convert raw data into things we handle
        self.cable_props = A.neuron_cable_properties()
        properties = read_dict(data, "cable_cell_globals")
        self.cable_props.catalogue.extend(A.allen_catalogue(), "")
        if properties:
            self.cable_props.set_property(
                Vm=properties["v_init"] * U.mV, tempK=properties["celsius"] * U.Celsius
            )
        max_extent = read_float(data, "max_cv_length")
        if max_extent:
            self.cv_policy = A.cv_policy_max_extent(max_extent)
        else:
            self.cv_policy = A.cv_policy_single()
        # clean-up...
        close_sim(data)
        # caches some data
        self.cable_data = {}

    def cell_kind(self, gid):
        kind = self.gid_to_kid[gid]
        if kind == 0:
            return A.cell_kind.cable
        elif kind == 1:
            return A.cell_kind.lif
        elif kind == 2:
            return A.cell_kind.spike_source
        else:
            raise RuntimeError("Unknown cell kind")

    def num_cells(self):
        return self.N

    def connections_on(self, gid):
        if not gid in self.gid_to_inc:
            return []
        return [
            A.connection((src, lbl), tgt, w, max(d, self.dt) * U.ms)
            for src, lbl, tgt, w, d in self.gid_to_inc[gid]
        ]

    def global_properties(self, kind):
        if kind == A.cell_kind.cable:
            return self.cable_props
        return None

    def cell_description(self, gid):
        kind = self.gid_to_kid[gid]
        if kind == 0:
            return self.make_cable_cell(gid)
        elif kind == 1:
            return self.make_lif_cell(gid)
        elif kind == 2:
            return self.make_vrt_cell(gid)
        else:
            raise RuntimeError("Unknown cell kind")

    def probes(self, gid):
        res = []
        if gid in self.gid_to_prb:
            kind = self.cell_kind(gid)
            for loc, var, tag in self.gid_to_prb[gid]:
                if kind == A.cell_kind.cable:
                    loc = f'(on-components 0.5 (region "{loc}"))'
                    if var == "voltage":
                        res.append(A.cable_probe_membrane_voltage(loc, tag))
                    elif var.endswith("i"):
                        res.append(
                            A.cable_probe_ion_int_concentration(loc, var[:-1], tag)
                        )
                    elif var.endswith("o"):
                        res.append(
                            A.cable_probe_ion_ext_concentration(loc, var[:-1], tag)
                        )
                    else:
                        print(f"[UNSUPPORTED] Skipping cable probe: {var}.")
                elif kind == A.cell_kind.lif:
                    if var == "voltage":
                        res.append(A.lif_probe_voltage(tag))
                    else:
                        raise RuntimeError(f"Probing var={var} not yet implemented")
                else:
                    raise RuntimeError(f"Probing cell of kind={kind} not implemented")
        return res

    def make_cable_cell(self, gid):
        mrf, dec = self.load_cable_data(gid)
        lbl = A.label_dict().add_swc_tags()
        # NOTE in theory we could have more and in other places...
        dec.place(
            "(location 0 0.5)", A.threshold_detector(self.threshold * U.mV), "src"
        )
        if gid in self.gid_to_syn:
            for location, synapse, params, tag in self.gid_to_syn[gid]:
                dec.place(location, A.synapse(synapse, **params), tag)
        if gid in self.gid_to_icp:
            for loc, delay, duration, amplitude, tag in self.gid_to_icp[gid]:
                dec.place(
                    loc,
                    A.iclamp(
                        tstart=delay * U.ms,
                        duration=duration * U.ms,
                        current=amplitude * U.nA,
                    ),
                    tag,
                )
        dec.discretization(self.cv_policy)

        return A.cable_cell(mrf, dec, lbl)

    def make_lif_cell(self, gid):
        cell = A.lif_cell("src", "tgt")
        data = self.gid_to_lif[gid]
        # setup the cell to adhere to NEURON's defaults
        cell.C_m = 0.6 * data["cm"] * U.pF
        cell.tau_m = data["tau"] * U.ms
        cell.E_L = data["U_neutral"] * U.mV
        cell.E_R = data["U_reset"] * U.mV
        cell.V_m = data["U_0"] * U.mV
        cell.V_th = data["U_th"] * U.mV
        cell.t_ref = data["t_ref"] * U.ms
        return cell

    def make_vrt_cell(self, gid):
        return A.spike_source_cell(
            "src", A.explicit_schedule([t * U.ms for t in self.gid_to_vrt[gid]])
        )

    def load_cable_data(self, gid):
        mid, cid = self.gid_to_bio[gid]
        if not gid in self.cable_data:
            timing.tic("build/simulation/io")
            mrf = load_morphology(here / "mrf" / self.mid_to_mrf[mid])
            dec = A.load_component(here / "acc" / self.cid_to_acc[cid]).component
            self.cable_data[gid] = (mrf, dec)
            timing.toc("build/simulation/io")
        mrf, dec = self.cable_data[gid]
        return mrf, A.decor(dec)  # NOTE copy that decor!!


timing.tic("build/recipe")
rec = recipe()
timing.toc("build/recipe")

timing.tic("build/simulation")
ctx = A.context()
hints = {}
for kind, tag in zip(
    [A.cell_kind.cable, A.cell_kind.lif, A.cell_kind.spike_source], [0, 1, 2]
):
    if tag in rec.kind_to_count:
        hints[kind] = A.partition_hint(
            cpu_group_size=rec.kind_to_count[tag] / ctx.threads
        )
ddc = A.partition_load_balance(rec, ctx, hints)
sim = A.simulation(rec, context=ctx, domains=ddc)
timing.toc("build/simulation")

timing.tic("build/sampling")
sim.record(A.spike_recording.all)

schedule = A.regular_schedule(tstart=0 * U.ms, dt=10 * rec.dt * U.ms)
handles = {
    (gid, tag): sim.sample((gid, tag), schedule=schedule)
    for gid, prbs in rec.gid_to_prb.items()
    for _, _, tag in prbs
}
timing.toc("build/sampling")

timing.tic("run")
sim.run(rec.T * U.ms, rec.dt * U.ms)
timing.toc("run")

timing.tic("output/spikes")
spikes = sim.spikes()
df = pd.DataFrame(
    {
        "time": spikes["time"],
        "gid": spikes["source"]["gid"],
        "lid": spikes["source"]["index"],
    }
)
df["kind"] = df["gid"].map(lambda i: rec.gid_to_kid[i])
df["population"] = df["gid"].map(lambda i: rec.gid_to_meta[i]["population"])
df["type"] = df["gid"].map(lambda i: rec.gid_to_meta[i]["type_id"])
df.to_csv(here / "out" / "spikes.csv")
timing.toc("output/spikes")

timing.tic("output/samples")
for (gid, tag), handle in handles.items():
    dfs = []
    for data, meta in sim.samples(handle):
        if isinstance(meta, list):
            columns = list(map(str, meta))
        else:
            columns = [str(meta)]
        assert data.shape[1] == len(columns) + 1
        dfs.append(pd.DataFrame(data=data[:, 1:], columns=columns, index=data[:, 0]))
    if not dfs:
        print(f"[WARN] No data collected for tag '{tag}' on cell {gid}")
        continue
    df = pd.concat(dfs, axis=1)
    df.index.name = "t/ms"
    df.to_csv(here / "out" / f"gid_{gid}-tag_{tag}.csv")

    fg, ax = plt.subplots()
    df.plot(ax=ax)
    fg.savefig(here / "out" / f"gid_{gid}-tag_{tag}.pdf")
timing.toc("output/samples")

if have_stats:
    timing.tic("output/statistics")
    N = rec.num_cells()

    cells = defaultdict(lambda: defaultdict(lambda: 0))
    spike = defaultdict(lambda: defaultdict(lambda: 0))
    conns = defaultdict(lambda: 0)

    for gid in range(N):
        meta = rec.gid_to_meta[gid]
        pop = meta["population"]
        kind = meta["type_id"]
        cells[pop][kind] += 1
        cells[pop][-1] += 1
        for conn in rec.connections_on(gid):
            src = rec.gid_to_meta[conn.source.gid]["population"]
            conns[(src, pop)] += 1

    for (gid, _), _ in spikes:
        meta = rec.gid_to_meta[gid]
        pop = meta["population"]
        kind = meta["type_id"]
        spike[pop][kind] += 1
        spike[pop][-1] += 1
    C = sum(conns.values())
    timing.toc("output/statistics")

    print(
        f"""
Statistics
==========

* Cells                  {N:>13}"""
    )
    for pop, kinds in cells.items():
        print(f"  * {pop:<20} {kinds[-1]:>13}")
        for kind, num in kinds.items():
            if kind == -1:
                continue
            print(f"    * {kind:<18} {num:>13}")
    print(f"* Connections            {C:>13}")
    for (src, tgt), num in conns.items():
        print(f"  * {src:<8} -> {tgt:<8} {num:>13}")
    print(f"* Spikes                 {len(spikes):>13}")
    for pop, kinds in spike.items():
        print(f"  * {pop:<20} {kinds[-1]:>13}")
        for kind, num in kinds.items():
            if kind == -1:
                continue
            print(f"    * {kind:<18} {num:>13}")


def show_times(root, childs, time, prefix):
    lbl = f"{' '*prefix}* {root}"
    print(f"{lbl:<37}{time[root]:0.3f}")
    for child in childs[root]:
        show_times(child, childs, time, prefix + 2)


if have_timing:
    times = defaultdict(lambda: 0.0)
    children = defaultdict(set)
    for path, time in timing.timings.items():
        last = "Total"
        times[last] += time
        for k in path.split("/"):
            children[last].add(k)
            last = k
            times[k] += time

    print(
        """
Timings
==========
    """
    )
    show_times("Total", children, times, 0)
