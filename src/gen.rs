use crate::{
    err::Result,
    fit::Attribute,
    sim::{CVPolicy, IClamp, ModelType, Probe, Simulation, Edge, Node},
    Map,
};
use anyhow::bail;
use serde::Serialize;

/// Resources to store in the output.

type ConnectionData = (usize, String, String, f64, f64);
type ProbeData = (Option<String>, String, String);
type SynapseData = (String, String, Map<String, f64>, String);
type IClampData = (String, f64, f64, f64, String);

/// Metadata about cell,
/// mostly info discarded during generation
#[derive(Debug, Serialize)]
pub struct CellMetaData {
    /// cell kind
    pub kind: String,
    /// population name
    pub population: String,
    /// cell type label
    pub type_id: u64,

}

impl CellMetaData {
    pub fn from(node: &Node) -> Self {
        let kind = match node.node_type.model_type {
            ModelType::Biophysical { .. } => String::from("biophys"),
            ModelType::Single { .. } => String::from("single"),
            ModelType::Point { .. } => String::from("point"),
            ModelType::Virtual { .. } => String::from("virtual"),
        };
        Self {
            population: node.pop.to_string(),
            kind,
            type_id: node.node_type.type_id
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Bundle {
    pub time: f64,
    pub time_step: f64,
    pub size: usize,
    pub max_cv_length: Option<f64>,
    /// gid to morphology and acc ids
    /// this works as an index into the next two fields.
    pub cell_bio_ids: Map<usize, (usize, usize)>,
    /// acc index to name
    pub morphology: Vec<String>,
    /// morphology index to name
    pub decoration: Vec<String>,
    /// cell kinds, 0 = cable, 1 = lif, 2 = spike source, ...
    pub cell_kind: Vec<u64>,
    /// synapse data, cross-linked with incoming connections.
    /// Location, Synapse, Parameters, Tag
    /// May only be set iff kind==cable
    pub synapses: Map<usize, Vec<SynapseData>>,
    /// stimuli; May only be set iff kind==cable
    /// location, delay, duration, amplitude, tag
    pub current_clamps: Map<usize, Vec<IClampData>>,
    /// List of data exporters
    /// location, variable, tag. NOTE _could_ make variable an u64?!
    pub probes: Map<usize, Vec<ProbeData>>,
    /// Incoming connections as (src_gid, src_tag, tgt_tag, weight, delay)
    pub incoming_connections: Map<usize, Vec<ConnectionData>>,
    /// Spiking threshold
    pub spike_threshold: f64,
    /// sparse map of gids to LIF cell descrption. Valid iff kind(gid) == LIF
    pub gid_to_lif: Map<usize, Map<String, f64>>,
    /// sparse map of gids to virtual cell spike trains. Valid iff kind(gid) == Virtual
    /// Will generate SpikeSource cells in Arbor
    pub gid_to_vrt: Map<usize, Vec<f64>>,
    /// dense map of gids to metadata
    pub metadata: Vec<CellMetaData>,
}

const KIND_CABLE: u64 = 0;
const KIND_LIF: u64 = 1;
const KIND_SOURCE: u64 = 2;

fn fix_edge(edge: &Edge) -> Result<Edge> {
    if edge.mech.is_none() {
        bail!("Edge requires associated dynamics, but we found none.");
    }
    let mut edge = edge.clone();
    let mech = edge.mech.as_ref().unwrap();
    // if mech == "Exp2Syn" {
        // edge.mech = Some("exp2syn".to_string());
        // if let Some(v) = edge.dynamics.get("erev") {
            // edge.dynamics.insert("e".to_string(), *v);
            // edge.dynamics.remove("erev");
        // }
    // }
    Ok(edge)
}

impl Bundle {
    pub fn new(sim: &Simulation) -> Result<Self> {
        // Reverse lookup tables, used internally for uniqueness and index generation.
        let mut acc_to_cid = Map::new();
        let mut mrf_to_mid = Map::new();

        // Look up tables to write out
        let mut gid_to_meta = Vec::new();
        let mut cell_bio_ids = Map::new();
        let mut morphology = Vec::new();
        let mut decoration = Vec::new();
        let mut cell_kind = Vec::new();
        let mut incoming_connections = Map::new();
        let mut synapses = Map::new();
        let mut current_clamps = Map::new();
        let mut probes = Map::new();
        let mut gid_to_lif = Map::new();
        let mut gid_to_vrt = Map::new();
        for gid in 0..sim.size {
            let node = sim.reify_node(gid)?;
            gid_to_meta.push(CellMetaData::from(&node));
            if !node.incoming_edges.is_empty() {
                if matches!(node.node_type.model_type, ModelType::Biophysical { .. }) {
                    let mut inc = Vec::new();
                    let mut syn = Vec::new();
                    for (ix, edge) in node.incoming_edges.iter().enumerate() {
                        let tgt = format!("syn_{ix}");
                        inc.push((
                            edge.src_gid as usize,
                            String::from("src"), // in our SONATA model, there is _one_ source on each cell.
                            tgt.clone(),
                            edge.weight,
                            edge.delay,
                        ));
                        let edge = fix_edge(edge)?;
                        syn.push((
                            String::from("(location 0 0.5)"),
                            edge.mech.unwrap().to_string(),
                            edge.dynamics.clone(),
                            tgt,
                        ));
                    }
                    incoming_connections.insert(gid, inc);
                    synapses.insert(gid, syn);
                } else {
                    let inc = node
                        .incoming_edges
                        .iter()
                        .map(|e| {
                            (
                                e.src_gid as usize,
                                String::from("src"), // in our SONATA model, there is _one_ source on each cell.
                                String::from("tgt"),
                                e.weight,
                                e.delay,
                            )
                        })
                        .collect::<Vec<_>>();
                    incoming_connections.insert(gid, inc);
                };
            }

            match &node.node_type.model_type {
                ModelType::Biophysical {
                    model_template,
                    attributes,
                } => {
                    cell_kind.push(KIND_CABLE);
                    match model_template.as_ref() {
                        "ctdb:Biophys1.hoc" => {
                            let mid = if let Some(Attribute::String(mrf)) =
                                attributes.get("morphology")
                            {
                                if !mrf_to_mid.contains_key(mrf) {
                                    let mid = morphology.len();
                                    morphology.push(mrf.to_string());
                                    mrf_to_mid.insert(mrf.to_string(), mid);
                                }
                                mrf_to_mid[mrf]
                            } else {
                                bail!("GID {gid} is a biophysical cell, but has no morphology.");
                            };
                            let cid = if let Some(Attribute::String(fit)) =
                                attributes.get("dynamics_params")
                            {
                                let acc = fit;
                                if !acc_to_cid.contains_key(acc) {
                                    let cid = decoration.len();
                                    decoration.push(acc.to_string());
                                    acc_to_cid.insert(acc.to_string(), cid);
                                }
                                acc_to_cid[acc]
                            } else {
                                bail!(
                                    "GID {gid} is a biophysical cell, but has no dynamics_params."
                                );
                            };
                            cell_bio_ids.insert(gid, (mid, cid));
                        }
                        t => bail!("Unknown model template <{t}> for gid {gid}"),
                    }
                }
                ModelType::Virtual { .. } => { // The fields are largely irrelevant here
                    let data: &mut Vec<f64> = gid_to_vrt.entry(gid).or_default();
                    if let Some(group) = sim.virtual_spikes.get(&node.pop) {

                        if let Some(ts) = group.get(&node.node_id) {
                            data.append(&mut ts.clone());
                        }
                    }
                    cell_kind.push(KIND_SOURCE);
                }
                ModelType::Point { model_template, .. } => {
                    cell_kind.push(KIND_LIF);
                    match model_template.as_ref() {
                        "nrn:IntFire1" => {
                            // Taken from nrn/IntFire1.mod and adapted to Arbor.
                            let mut params = Map::from([("cm".to_string(), 1.0),
                                                        ("U_neutral".to_string(), 0.0),
                                                        ("U_reset".to_string(), 0.0),
                                                        ("U_th".to_string(), 1.0), // TODO IntFire1 do be weird.
                                                        ("U_0".to_string(), 0.0),
                                                        ("t_ref".to_string(), 5.0),
                                                        ("tau".to_string(), 10.0),]);
                            for (k, v) in node.dynamics.iter() {
                                match k.as_ref() {
                                    "tau" =>
                                        params.insert("tau".to_string(), *v),
                                    "refrac" =>
                                        params.insert("t_ref".to_string(), *v),
                                    _ => bail!("Unknown parameter <{k}> for template IntFire1 at gid {gid}")
                                };
                            }
                            gid_to_lif.insert(gid, params);
                        }
                        t => bail!("Unknown model template <{t}> for gid {gid}"),
                    }
                }
                mt => bail!("Cannot write ModelType {mt:?}"),
            }
        }

        for (gid, ics) in &sim.iclamps {
            let mut stim = Vec::new();
            for IClamp {
                amplitude_nA,
                delay_ms,
                duration_ms,
                tag,
                location,
            } in ics
            {
                stim.push((
                    location.clone(),
                    *delay_ms,
                    *duration_ms,
                    *amplitude_nA,
                    tag.clone(),
                ));
            }
            current_clamps.insert(*gid as usize, stim);
        }

        for (gid, sim_probes) in &sim.reports {
            let mut prbs = Vec::new();
            for probe in sim_probes {
                match probe {
                    Probe::CableVoltage(ls) => {
                        for l in ls {
                            prbs.push((Some(l.clone()), "voltage".into(), format!("prb-voltage@{l}")));
                        }
                    }
                    Probe::Lif => {
                        prbs.push((None, "voltage".into(), "prb-voltage".into()));
                    }
                    Probe::CableIntConc(ion, ls) => {
                        for l in ls {
                            prbs.push((Some(l.clone()), ion.clone(), format!("prb-{ion}@{l}")));
                        }
                    }
                    Probe::CableState(var, ls) => {
                        for l in ls {
                            prbs.push((Some(l.clone()), var.clone(), format!("prb-{var}@{l}")));
                        }
                    }
                    Probe::CableExtConc(ion, ls) => {
                        for l in ls {
                            prbs.push((Some(l.clone()), ion.clone(), format!("prb-{ion}@{l}")));
                        }
                    }
                }
            }
            probes.insert(*gid as usize, prbs);
        }

        let max_cv_length = match &sim.cv_policy {
            CVPolicy::Default => None,
            CVPolicy::MaxExtent(l) => Some(*l),
        };

        Ok(Bundle {
            time: sim.tfinal,
            time_step: sim.dt,
            max_cv_length,
            size: sim.size,
            cell_bio_ids,
            morphology,
            decoration,
            synapses,
            probes,
            incoming_connections,
            cell_kind,
            current_clamps,
            spike_threshold: sim.spike_threshold,
            gid_to_lif,
            gid_to_vrt,
            metadata: gid_to_meta,
        })
    }
}
