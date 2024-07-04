#!/usr/bin/env python3

from collections import defaultdict
import arbor as A
from arbor import units as U
import pandas as pd
from json import load as load_data
import matplotlib.pyplot as plt
from time import perf_counter as pc

from pathlib import Path

here = Path(__file__).parent

def load_morphology(path):
    sfx = path.suffix
    if sfx == '.swc':
        try:
            return A.load_swc_arbor(path).morphology
        except:
            pass
        try:
            return A.load_swc_neuron(path).morphology
        except:
            raise RuntimeError(f"Could load {path} neither as NEURON nor Arbor flavour.")
    elif sfx == '.asc':
        return A.load_asc(path).morphology
    elif sfx == '.nml':
        nml = A.load_nml(path)
        if len(nml.morphology_ids()) == 1:
            return nml.morphology(nml.morphology_ids()[0]).morphology
        else:
            raise RuntimeError(f"NML file {path} contains multiple morphologies.")
    else:
        raise RuntimeError(f"Unknown morphology file type {sfx}")


class recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)
        # Global cable cell settings
        self.cv_policy = A.cv_policy_single()
        self.cable_props = A.neuron_cable_properties()
        self.cable_props.catalogue.extend(A.allen_catalogue(), '')
        with open(here / 'dat/sim.json', 'rb') as fd:
            data = load_data(fd)
            # Some initial, global data
            self.N = data['size']
            self.T = data['time']
            self.dt = data['time_step']
            self.threshold = data['spike_threshold']
            if data['max_cv_length']:
                self.cv_policy = A.cv_policy_max_extent(data['max_cv_length'])
            # gid -> (morphology id, acc id)
            self.gid_to_bio = { int(k): v for k, v in data['cell_bio_ids'].items() }
            # gid -> lif cell data
            self.gid_to_lif = { int(k): v for k, v in data['gid_to_lif'].items() }
            # gid -> virtual cell data
            self.gid_to_vrt = { int(k): v for k, v in data['gid_to_vrt'].items() }
            # morphology id -> morphology resource file
            self.mid_to_mrf = data['morphology']
            # cell id -> decor file
            self.cid_to_acc = data['decoration']
            # gid -> cell kind
            self.gid_to_kid = data['cell_kind']
            # gid -> cell metadata
            self.gid_to_meta = data['metadata']
            # gid -> incoming connections
            self.gid_to_inc = { int(k): v for k, v in data['incoming_connections'].items() }
            # gid -> synapse
            self.gid_to_syn = { int(k): v for k, v in data['synapses'].items() }
            # gid -> iclamps. NOTE must only be set if kind[gid] == cable
            self.gid_to_icp = { int(k): v for k, v in data['current_clamps'].items() }
            # probes
            self.gid_to_prb = { int(k): v for k, v in data['probes'].items() }
            globals = data['cable_cell_globals']
            self.kind_to_count = data['count_by_kind']
            if globals:
                self.cable_props.set_property(Vm=globals["v_init"] * U.mV, tempK=globals["celsius"] * U.Celsius)
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
            raise RuntimeError('Unknown cell kind')

    def num_cells(self):
        return self.N

    def connections_on(self, gid):
        if not gid in self.gid_to_inc:
            return []
        return [A.connection((src, lbl), tgt, w, max(d, self.dt) * U.ms)
                for src, lbl, tgt, w, d
                in self.gid_to_inc[gid]]

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
            raise RuntimeError('Unknown cell kind')

    def probes(self, gid):
        res = []
        if gid in self.gid_to_prb:
            kind = self.cell_kind(gid)
            for loc, var, tag in self.gid_to_prb[gid]:
                if kind == A.cell_kind.cable:
                    loc = f'(on-components 0.5 (region "{loc}"))'
                    if var == 'voltage':
                        res.append(A.cable_probe_membrane_voltage(loc, tag))
                    elif var.endswith('i'):
                        res.append(A.cable_probe_ion_int_concentration(loc, var[:-1], tag))
                    elif var.endswith('o'):
                        res.append(A.cable_probe_ion_ext_concentration(loc, var[:-1], tag))
                    else:
                        print(f"Skipping unknown cable probe: {var}")
                elif kind == A.cell_kind.lif:
                    if var == 'voltage':
                        res.append(A.lif_probe_voltage(tag))
                    else:
                        raise RuntimeError(f'Probing var={var} not yet implemented')
                else:
                    raise RuntimeError(f'Probing cell of kind={kind} not implemented')
        return res

    def make_cable_cell(self, gid):
        mrf, dec = self.load_cable_data(gid)
        lbl = A.label_dict().add_swc_tags()
        # NOTE in theory we could have more and in other places...
        dec.place('(location 0 0.5)', A.threshold_detector(self.threshold * U.mV), 'src')
        if gid in self.gid_to_syn:
            for location, synapse, params, tag in self.gid_to_syn[gid]:
                dec.place(location, A.synapse(synapse, **params), tag)
        if gid in self.gid_to_icp:
            for loc, delay, duration, amplitude, tag  in self.gid_to_icp[gid]:
                dec.place(loc, A.iclamp(tstart=delay * U.ms, duration=duration * U.ms, current=amplitude * U.nA), tag)
        dec.discretization(self.cv_policy)

        return A.cable_cell(mrf, dec, lbl)

    def make_lif_cell(self, gid):
        cell = A.lif_cell('src', 'tgt')
        data = self.gid_to_lif[gid]
        # setup the cell to adhere to NEURON's defaults
        cell.C_m = 0.6*data['cm'] * U.pF
        cell.tau_m = data['tau'] * U.ms
        cell.E_L = data['U_neutral'] * U.mV
        cell.E_R = data['U_reset'] * U.mV
        cell.V_m = data['U_0'] * U.mV
        cell.V_th = data['U_th'] * U.mV
        cell.t_ref = data['t_ref'] * U.ms
        return cell

    def make_vrt_cell(self, gid):
        return A.spike_source_cell("src",
                                   A.explicit_schedule([t * U.ms for t in self.gid_to_vrt[gid]]))

    def load_cable_data(self, gid):
        mid, cid = self.gid_to_bio[gid]
        if not gid in self.cable_data:
            t0 = pc()
            mrf = load_morphology(here / 'mrf' / self.mid_to_mrf[mid])
            dec = A.load_component(here / 'acc' / self.cid_to_acc[cid]).component
            self.cable_data[gid] = (mrf, dec)
            t1 = pc()
            self.io += t1 - t0
        mrf, dec = self.cable_data[gid]
        return mrf, A.decor(dec) # NOTE copy that decor!!

t0 = pc()
rec = recipe()
t1 = pc()
ctx = A.context()
print(ctx)
hints = {}
for kind, tag in zip([A.cell_kind.cable, A.cell_kind.lif, A.cell_kind.spike_source], [0, 1, 2]):
    hints[kind] = A.partition_hint(cpu_group_size=rec.kind_to_count[tag]/ctx.threads)
ddc = A.partition_load_balance(rec, ctx, hints)
print(ctx)
print(ddc)
sim = A.simulation(rec, context=ctx, domains=ddc)
t2 = pc()
sim.record(A.spike_recording.all)

schedule = A.regular_schedule(tstart=0*U.ms, dt=10*rec.dt*U.ms)
handles = { (gid, tag): sim.sample((gid, tag), schedule=schedule)
            for gid, prbs in rec.gid_to_prb.items()
            for _, _, tag in prbs}
t3 = pc()
sim.run(rec.T*U.ms, rec.dt*U.ms)
t4 = pc()
spikes = sim.spikes()
df = pd.DataFrame({"time": spikes["time"],
                   "gid": spikes['source']['gid'],
                   "lid": spikes['source']['index']})
df['kind'] = df['gid'].map(lambda i: rec.gid_to_kid[i])
df['population'] = df['gid'].map(lambda i: rec.gid_to_meta[i]["population"])
df['type'] = df['gid'].map(lambda i: rec.gid_to_meta[i]["type_id"])
df.to_csv(here / 'out' / 'spikes.csv')
t5 = pc()
for (gid, tag), handle in handles.items():
    dfs = []
    for data, meta in sim.samples(handle):
        columns = []
        if isinstance(meta, list):
            columns += list(map(str, meta))
        else:
            columns.append(str(meta))
        dfs.append(pd.DataFrame(data=data[:, 1:], columns=columns, index=data[:, 0]))
    df = pd.concat(dfs, axis=1)
    df.index.name = 't/ms'
    df.to_csv(here / 'out' / f'gid_{gid}-tag_{tag}.csv')

    fg, ax = plt.subplots()
    df.plot(ax=ax)
    fg.savefig(here / 'out' / f'gid_{gid}-tag_{tag}.pdf')
t6 = pc()

N = rec.num_cells()

cells = defaultdict(lambda: defaultdict(lambda: 0))
spike = defaultdict(lambda: defaultdict(lambda: 0))
conns = defaultdict(lambda: 0)

for gid in range(N):
    meta = rec.gid_to_meta[gid]
    pop = meta['population']
    kind = meta['type_id']
    cells[pop][kind] += 1
    cells[pop][-1] += 1
    for conn in rec.connections_on(gid):
        src = rec.gid_to_meta[conn.source.gid]['population']
        conns[(src, pop)] += 1

for (gid, _), _ in spikes:
    meta = rec.gid_to_meta[gid]
    pop = meta['population']
    kind = meta['type_id']
    spike[pop][kind] += 1
    spike[pop][-1] += 1
C = sum(conns.values())
t7 = pc()

print(f"""
Statistics
==========

* Cells                  {N:>13}""")
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
print(f"""* Runtime                {t6 - t0:>13.3f}s
  * building             {t3 - t0:>13.3f}s
    * recipe             {t1 - t0:>13.3f}s
    * simulation         {t2 - t1:>13.3f}s
      * reading data     {rec.io:>13.3f}s
    * sampling           {t3 - t2:>13.3f}s
  * run                  {t4 - t3:>13.3f}s
  * output               {t6 - t4:>13.3f}s
    * spikes             {t5 - t4:>13.3f}s
    * samples            {t6 - t5:>13.3f}s
  * statistics           {t7 - t6:>13.3f}s
""")
