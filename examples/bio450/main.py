#!/usr/bin/env python3

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
        self.cv_policy = A.cv_policy_single()
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
            # gid -> incoming connections
            self.gid_to_inc = { int(k): v for k, v in data['incoming_connections'].items() }
            # gid -> synapse
            self.gid_to_syn = { int(k): v for k, v in data['synapses'].items() }
            # gid -> iclamps. NOTE must only be set if kind[gid] == cable
            self.gid_to_icp = { int(k): v for k, v in data['current_clamps'].items() }
            # probes
            self.gid_to_prb = { int(k): v for k, v in data['probes'].items() }
        # Global cable cell settings
        self.cable_props = A.neuron_cable_properties()
        self.cable_props.catalogue.extend(A.allen_catalogue(), '')

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
        mid, cid = self.gid_to_bio[gid]
        mrf = load_morphology(here / 'mrf' / self.mid_to_mrf[mid])
        acc = self.cid_to_acc[cid]
        dec = A.load_component(here / 'acc' / acc).component
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
        cell.C_m = data['cm'] * U.pF
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

t0 = pc()
rec = recipe()
t1 = pc()
sim = A.simulation(rec)
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
print(f"Stats recipe={t1 - t0:.3}s simulation={t2 - t1:.3}s sampling={t3 - t2:.3}s run={t4 - t3:.3}s spikes={t5 - t4:.3}s samples={t6 - t5:.3}s")
