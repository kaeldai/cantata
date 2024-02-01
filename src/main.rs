use anyhow::{anyhow, bail};
use clap::{self, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sonata::{
    err::{Context, Result},
    raw, Map,
    fit::{Fit, self, Parameter},
};

use std::{fs::File, str::FromStr};

#[derive(Parser)]
#[clap(name = "sonata")]
#[clap(version = "0.1.0-dev", author = "t.hater@fz-juelich.de")]
struct Cli {
    #[clap(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Build { from: String, to: String },
    Fit { from: String },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum Attribute {
    String(String),
    Float(f64),
}

/// Model types currently known to SONATA
///
/// * single_compartment
///
///   A single cylindrical compartment is created with length equal to diameter,
///   thus the same effective area as that of a sphere of the same diameter. The
///   diameter is defined by an additional expected dynamics_param `D`, which
///   defaults to 1 um. Further, the passive mechanism is inserted and the
///   additional mechanisms named in the `model_template` required attribute.
///
/// * point_neuron
///
///   The actual model type is defined by the `model_template` required
///   attribute, eg an NMODL file for NRN and for NEST/PyNN, model_template will
///   provide the name of a built-in model.
///
/// * virtual
///
///   Placeholder neuron, which is not otherwise simulated, but can a the source
///   of spikes.
///
/// * biophysical
///
///   A compartmental neuron. The attribute morphology must be defined, either
///   via the node or node_type.
///
///  In addition, any number of Key:Value pairs may be listed
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "model_type")]
pub enum ModelType {
    #[serde(rename = "biophysical")]
    Biophysical {
        model_template: String,
        #[serde(flatten)]
        attributes: Map<String, Attribute>,
    },
    #[serde(rename = "single_compartment")]
    Single {
        model_template: String,
        #[serde(flatten)]
        attributes: Map<String, Attribute>,
    },
    #[serde(rename = "point_neuron")]
    Point {
        model_template: String,
        #[serde(flatten)]
        attributes: Map<String, Attribute>,
    },
    #[serde(rename = "virtual")]
    Virtual {
        #[serde(flatten)]
        attributes: Map<String, Attribute>,
    },
}

/// Types are defined in CSV files with one named column for each attribute.
/// Separator is a single space.
/// Non scalar attributes may be given provided values are quoted and their
/// components are separated by spaces.
/// Node type columns will be assigned to node attributes indexed by the type id.
/// Columns are:
/// - node_type_id: required; defines the node_type_id of this row.
/// - population: required in either this, or the nodes.h5; defines the population.
///               Multiple populations may define the same node_type_id,
/// - model_type: required, and may be defined only in the node_types.csv
/// - required columns may also appear in the instance H5; defined by the population
#[derive(Debug, Serialize, Deserialize, Clone)]
struct NodeType {
    /// unique -- within population -- id of this type. Used in instances to
    /// reference this type.
    #[serde(rename = "node_type_id")]
    type_id: u64,
    /// population using this type. Might be given by instance file, this, or
    /// both
    #[serde(rename = "pop_name")]
    population: Option<String>,
    /// holds the actual parameter data
    #[serde(flatten)]
    model_type: ModelType,
}

impl NodeType {
    fn clean_up(&mut self) {
        match &mut self.model_type {
            ModelType::Biophysical { attributes, .. }
            | ModelType::Single { attributes, .. }
            | ModelType::Point { attributes, .. }
            | ModelType::Virtual { attributes } => {
                attributes.retain(|_, v| !matches!(v, Attribute::String(s) if s == "NULL"))
            }
        }
    }
}

#[derive(Debug)]
struct ParameterGroup {
    id: u64,
    dynamics: Map<String, Vec<f64>>,
    custom: Map<String, Vec<f64>>,
}

/// Populations are stored HDF5 files, and have an associated node types file to
/// define node_type_ids and assign attributes common across a population.
/// NOTE: node_types file may be shared by multiple population files.
///
/// Node groups are represented as HDF5 groups (with population as parent)
/// containing a dataset for each parameter of length equal to the number of
/// nodes in the group.
///
/// If an attribute is defined in both the node types and the node instances, the
/// latter overrides the former.
///
/// We have the following layout for the node instance HDF5 file:
///
/// Path                                 Type                      Required
/// =================================    ======================    ============
/// /nodes                               Group
///     * /<population_name>             Group
///         * /node_type_id              Dataset{N_total_nodes}    X
///         * /node_id                   Dataset{N_total_nodes}    X
///         * /node_group_id             Dataset{N_total_nodes}    X
///         * /node_group_index          Dataset{N_total_nodes}    X
///         * /<group_id>                Group                     one per unique group_id, at least one
///             * /dynamics_params       Group                     X (may be empty, though)
///                 * /<param>           Dataset {M_nodes}
///             * /<custom_attribute>    Dataset {M_nodes}
///
/// Notes:
/// * For each unique entry in node_group_id we expect one <group_id> group
///   under the population
#[derive(Debug)]
struct NodePopulation {
    name: String,
    size: usize,
    type_ids: Vec<u64>,
    node_ids: Vec<u64>,
    group_ids: Vec<u64>,
    group_indices: Vec<usize>,
    groups: Vec<ParameterGroup>,
}

#[derive(Debug)]
struct NodeList {
    types: Vec<NodeType>,
    populations: Vec<NodePopulation>,
    size: usize,
}

impl NodeList {
    fn new(nodes: &raw::Nodes) -> Result<Self> {
        let path = &nodes.types;
        let path = std::path::PathBuf::from_str(&nodes.types)
            .map_err(anyhow::Error::from)
            .and_then(|p| p.canonicalize().map_err(anyhow::Error::from))
            .with_context(|| format!("Resolving node types {path}"))?;
        let rd = File::open(&path).with_context(|| format!("Opening {path:?}"))?;
        let mut tys = csv::ReaderBuilder::new()
            .delimiter(b' ')
            .from_reader(rd)
            .deserialize()
            .map(|it| it.map_err(anyhow::Error::from))
            .collect::<Result<Vec<NodeType>>>()
            .with_context(|| format!("Parsing node types {path:?}"))?;
        tys.iter_mut().for_each(|nt| nt.clean_up());
        let path = &nodes.nodes;
        let path = std::path::PathBuf::from_str(&nodes.nodes)
            .map_err(anyhow::Error::from)
            .and_then(|p| p.canonicalize().map_err(anyhow::Error::from))
            .with_context(|| format!("Resolving node instances {path}"))?;
        let node_instance_file = hdf5::file::File::open(&path)?;
        let nodes = node_instance_file.group("nodes")?;
        let populations = nodes
            .groups()
            .with_context(|| "Opening population list".to_string())?;

        let mut total_size = 0;
        let mut pops = Vec::new();
        for population in &populations {
            let name = population.name().rsplit_once('/').unwrap().1.to_string();
            let type_ids = get_dataset::<u64>(population, "node_type_id")?;
            let size = type_ids.len();
            let node_ids = get_dataset::<u64>(population, "node_id")
                .unwrap_or_else(|_| (0..size as u64).collect::<Vec<_>>());
            if size != node_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #node_ids"
                )
            }
            let group_ids = get_dataset::<u64>(population, "node_group_id")?;
            if size != group_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #group_id"
                )
            }
            let group_indices = get_dataset::<usize>(population, "node_group_index")?;
            if size != group_indices.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #group_index"
                )
            }
            let mut groups = Vec::new();
            // NOTE We assume here that group ids are contiguous; yet, that is
            // said nowhere.
            let mut group_id = 0;
            loop {
                if let Ok(group) = population.group(&format!("{group_id}")) {
                    let mut dynamics = Map::new();
                    let mut custom = Map::new();
                    if let Ok(dynamics_params) = group.group("dynamics_params") {
                        for param in dynamics_params.datasets()?.iter() {
                            let values = param.read_1d::<f64>()?.to_vec();
                            let name = param.name().rsplit_once('/').unwrap().1.to_string();
                            dynamics.insert(name, values);
                        }
                    }
                    for param in group.datasets()?.iter() {
                        let values = param.read_1d::<f64>()?.to_vec();
                        let name = param.name().rsplit_once('/').unwrap().1.to_string();
                        custom.insert(name, values);
                    }

                    groups.push(ParameterGroup {
                        id: group_id as u64,
                        dynamics,
                        custom,
                    });
                } else {
                    break;
                }
                group_id += 1;
            }
            total_size += size;
            pops.push(NodePopulation {
                name,
                size,
                type_ids,
                node_ids,
                group_ids,
                group_indices,
                groups,
            })
        }
        Ok(Self {
            types: tys,
            populations: pops,
            size: total_size,
        })
    }
}

/// types are defined in a CSV file of named columns; separator is a single space.
/// - edge_type_id; required
/// - population; required either in CSV or H5; handles populations defining the same edge_type_id
/// - any number of additional columns may freely be added.
#[derive(Debug, Serialize, Deserialize)]
struct EdgeType {
    #[serde(rename = "edge_type_id")]
    type_id: u64,
    #[serde(rename = "pop_name")]
    population: Option<String>,
    #[serde(flatten)]
    attributes: Map<String, Attribute>,
}

impl EdgeType {
    fn clean_up(&mut self) {
        self.attributes
            .retain(|_, v| !matches!(v, Attribute::String(s) if s == "NULL"));
    }
}

/// Populations are stored HDF5 files, and have an associated edge types file to
/// define edge_type_ids and assign attributes common across a population.
/// NOTE: edge_types file may be shared by multiple population files.
///
/// Edge groups are represented as HDF5 groups (with population as parent)
/// containing a dataset for each parameter of length equal to the number of
/// edges in the group.
///
/// Edge populations are defined as groups and stored as sparse table via the
/// `source_node_id` and `target_node_id` datasets. These datasets have an
/// associated attribute "node_population" that specifies the node population
/// for resolving the node_ids of the source or target.
///
/// If an attribute is defined in both the edge types and the edge instances, the
/// latter overrides the former.
///
/// We have the following layout for the edge instance HDF5 file:
///
/// Path                                 Type                      Required
/// =================================    ======================    ============
/// /edges                               Group
///     * /<population_name>             Group
///         * /edge_type_id              Dataset{N_total_edges}    X
///         * /edge_id                   Dataset{N_total_edges}    X
///         * /edge_group_id             Dataset{N_total_edges}    X
///         * /edge_group_index          Dataset{N_total_edges}    X
///         * /source_node_id            Dataset{N_total_edges}    X
///             * /population_name       Attribute                 X
///         * /target_node_id            Dataset{N_total_edges}    X
///             * /population_name       Attribute                 X
///         * /<group_id>                Group
///             * /dynamics_params       Group
///                 * /<param>           Dataset {M_edges}
///             * /<custom_attribute>    Dataset {M_edges}
///
/// Notes:
/// * For each unique entry in edge_group_id we expect one <group_id> group
///   under the population
#[derive(Debug)]
struct EdgePopulation {
    name: String,
    size: usize,
    type_ids: Vec<u64>,
    edge_ids: Vec<u64>,
    group_ids: Vec<u64>,
    group_indices: Vec<usize>,
    source_ids: Vec<u64>,
    source_pop: String,
    target_ids: Vec<u64>,
    target_pop: String,
    groups: Vec<ParameterGroup>,
}

#[derive(Debug)]
pub struct EdgeList {
    types: Vec<EdgeType>,
    populations: Vec<EdgePopulation>,
    /// Cumulative size of populations
    size: usize,
    /// dynamics parameters read from type CSV file; keyed on the filename(!)
    dynamics: Map<String, Map<String, f64>>,
}

fn get_dataset<T: hdf5::H5Type + Clone>(g: &hdf5::Group, nm: &str) -> Result<Vec<T>> {
    Ok(g.dataset(nm)
        .with_context(|| format!("Group {} has no dataset {nm}", g.name()))?
        .read_1d::<T>()?
        .to_vec())
}

impl EdgeList {
    fn new(edges: &raw::Edges) -> Result<Self> {
        let path = &edges.types;
        let path = std::path::PathBuf::from_str(&edges.types)
            .map_err(anyhow::Error::from)
            .and_then(|p| p.canonicalize().map_err(anyhow::Error::from))
            .with_context(|| format!("Resolving edge types {path}"))?;
        let rd = File::open(&path).with_context(|| format!("Opening {path:?}"))?;
        let mut tys = csv::ReaderBuilder::new()
            .delimiter(b' ')
            .from_reader(rd)
            .deserialize()
            .map(|it| it.map_err(anyhow::Error::from))
            .collect::<Result<Vec<EdgeType>>>()
            .with_context(|| format!("Parsing edge types {path:?}"))?;
        tys.iter_mut().for_each(|nt| nt.clean_up());
        let path = &edges.edges;
        let path = std::path::PathBuf::from_str(&edges.edges)
            .map_err(anyhow::Error::from)
            .and_then(|p| p.canonicalize().map_err(anyhow::Error::from))
            .with_context(|| format!("Resolving edge instances {path}"))?;
        let edge_instance_file = hdf5::file::File::open(&path)?;
        let edges = edge_instance_file.group("edges")?;
        let populations = edges
            .groups()
            .with_context(|| "Opening population list".to_string())?;

        let mut total_size = 0;
        let mut pops = Vec::new();
        for population in &populations {
            let name = population.name().rsplit_once('/').unwrap().1.to_string();
            let type_ids = get_dataset(population, "edge_type_id")?;
            let size = type_ids.len();
            let edge_ids = get_dataset::<u64>(population, "edge_id")
                .unwrap_or_else(|_| (0..size as u64).collect::<Vec<_>>());
            if size != edge_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #edge_ids"
                )
            }
            let group_ids = get_dataset::<u64>(population, "edge_group_id")?;
            if size != group_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #group_id"
                )
            }
            let group_indices = get_dataset::<usize>(population, "edge_group_index")?;
            if size != group_indices.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #group_index"
                )
            }
            let sources = population
                .dataset("source_node_id")
                .with_context(|| format!("Extracting source indices from population {name}"))?;
            let source_ids = sources.read_1d::<u64>()?.to_vec();
            if size != source_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #source_ids"
                )
            }
            let source_pop = sources
                .attr("node_population")
                .with_context(|| {
                    format!("Extracting source population from population {name}; not found")
                })?
                .read_scalar::<hdf5::types::VarLenUnicode>()
                .with_context(|| {
                    format!("Extracting source population from population {name}; not a string")
                })?
                .as_str()
                .to_string();
            let targets = population
                .dataset("target_node_id")
                .with_context(|| format!("Extracting target indices from population {name}"))?;

            let target_ids = targets.read_1d::<u64>()?.to_vec();
            let target_pop = targets
                .attr("node_population")
                .with_context(|| {
                    format!("Extracting target population from population {name}; not found")
                })?
                .read_scalar::<hdf5::types::VarLenUnicode>()
                .with_context(|| {
                    format!("Extracting target population from population {name}; not a string")
                })?
                .as_str()
                .to_string();
            if size != target_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #target_ids"
                )
            }
            let mut groups = Vec::new();
            let mut group_id = 0;
            loop {
                if let Ok(group) = population.group(&format!("{group_id}")) {
                    let mut dynamics = Map::new();
                    let mut custom = Map::new();
                    if let Ok(dynamics_params) = group.group("dynamics_params") {
                        for param in dynamics_params.datasets()?.iter() {
                            let values = param.read_1d::<f64>()?.to_vec();
                            let name = param.name().rsplit_once('/').unwrap().1.to_string();
                            dynamics.insert(name, values);
                        }
                    }
                    for param in group.datasets()?.iter() {
                        let values = param.read_1d::<f64>()?.to_vec();
                        let name = param.name().rsplit_once('/').unwrap().1.to_string();
                        custom.insert(name, values);
                    }

                    groups.push(ParameterGroup {
                        id: group_id,
                        dynamics,
                        custom,
                    });
                } else {
                    break;
                }
                group_id += 1;
            }
            total_size += size;
            pops.push(EdgePopulation {
                name,
                size,
                type_ids,
                edge_ids,
                group_ids,
                source_ids,
                target_ids,
                source_pop,
                target_pop,
                group_indices,
                groups,
            })
        }
        Ok(Self {
            types: tys,
            populations: pops,
            size: total_size,
            dynamics: Map::new(),
        })
    }
}

#[derive(Debug)]
pub struct Edge {
    pop: String,
    group_id: u64,
    index: usize,
    type_id: u64,
    src_gid: u64,
    mech: String,
    delay: f64,
    weight: f64,
    dynamics: Map<String, f64>,
}

/// Reified node, containing all information we currently have
#[derive(Debug)]
pub struct Node {
    /// globally (!) unique id
    gid: usize,
    /// owning population
    pop: String,
    /// id within the population
    node_id: u64,
    /// id of containing group.
    group_id: u64,
    /// index within the group
    group_index: usize,
    /// id of node type in population used to instantiate.
    node_type_id: u64,
    /// node type used to instantiate
    node_type: NodeType,
    /// connections terminating here.
    incoming_edges: Vec<Edge>,
    /// Dynamics parameters extracted from the population.
    dynamics: Map<String, f64>,
    /// Custom parameters extracted from the population.
    custom: Map<String, f64>,
}

/// Bookeeping: index into top-level structure, ie population `pop` is stored at
/// node_lists[pop.list_index].populations[pop.pop_index]. The cells in this
/// population have identifiers in the range [start, start + size)
#[derive(Debug)]
struct PopId {
    /// Name of the population
    name: String,
    /// Index into the containing {node, edge}_list
    list_index: usize,
    /// Index into the containing population list
    pop_index: usize,
    /// GID of first cell in this population
    start: usize,
    /// Number of cells in this population
    size: usize,
}

impl PopId {
    fn new_nodes(lid: usize, pid: usize, start: usize, pop: &NodePopulation) -> Result<Self> {
        Ok(PopId {
            name: pop.name.to_string(),
            list_index: lid,
            pop_index: pid,
            start,
            size: pop.size,
        })
    }
}

#[derive(Debug)]
pub struct Simulation {
    /// runtime
    tfinal: f64,
    /// timestep
    dt: f64,
    /// raw-ish node data
    node_lists: Vec<NodeList>,
    /// raw-ish edge data
    edge_lists: Vec<EdgeList>,
    /// GID from (population, id)
    population_to_gid: Map<(usize, u64), u64>,
    /// GID to   (population, id)
    gid_to_population: Vec<(usize, u64)>,
    /// node population list, to avoid copying strings and storing the indices
    /// into node_list and population.
    node_populations: Vec<PopId>,
    /// reverse mapping Name -> Id
    population_ids: Map<String, usize>,
    /// Number of total cells
    size: usize,
}

impl Simulation {
    pub fn new(sim: &raw::Simulation) -> Result<Self> {
        let node_lists = sim
            .network
            .nodes
            .iter()
            .map(NodeList::new)
            .collect::<Result<Vec<_>>>()?;
        let mut edge_lists = sim
            .network
            .edges
            .iter()
            .map(EdgeList::new)
            .collect::<Result<Vec<_>>>()?;

        for el in edge_lists.iter_mut() {
            for ty in el.types.iter_mut() {
                if let Some(Attribute::String(name)) = ty.attributes.get("dynamics_params") {
                    let fname = find_component(name, &sim.components)?;
                    let fdata = std::fs::read_to_string(fname)?;
                    let fdata: Map<String, serde_json::Value> = serde_json::from_str(&fdata).with_context(|| format!("Parsing JSON from {name}"))?;
                    let fdata = fdata.into_iter().filter_map(|(k, v)| v.as_f64().map(|v| (k.to_string(), v))).collect();
                    el.dynamics.insert(name.to_string(), fdata);
                }
            }
        }
        let mut gid = 0;
        let mut start = 0;
        let mut population_to_gid = Map::new();
        let mut population_ids = Map::new();
        let mut gid_to_population = Vec::new();
        let mut node_populations = Vec::new();
        for (lid, node_list) in node_lists.iter().enumerate() {
            for (pid, population) in node_list.populations.iter().enumerate() {
                let pop_idx = node_populations.len();
                node_populations.push(PopId::new_nodes(lid, pid, start, population)?);
                population_ids.insert(population.name.to_string(), pop_idx);
                for nid in &population.node_ids {
                    population_to_gid.insert((pop_idx, *nid), gid);
                    gid_to_population.push((pop_idx, *nid));
                    gid += 1;
                }
                start += population.size;
            }
        }

        Ok(Self {
            tfinal: sim.run.tstop,
            dt: sim.run.dt,
            node_lists,
            edge_lists,
            gid_to_population,
            population_to_gid,
            node_populations,
            population_ids,
            size: start,
        })
    }

    pub fn reify_node(&self, gid: usize) -> Result<Node> {
        let (pop_idx, node_id) = self
            .gid_to_population
            .get(gid)
            .ok_or_else(|| anyhow!("Unknown gid {gid}, must be in [0, {})", self.size))?;
        let pop_id = self.node_populations.get(*pop_idx).expect("");
        let node_list = &self.node_lists[pop_id.list_index];
        let population = &node_list.populations[pop_id.pop_index];
        let node_index = gid - pop_id.start;

        // store pre-built error for later
        let node_index_error = || {
            anyhow!(
                "Index {} overruns size {} of population {}",
                node_index,
                pop_id.size,
                population.name
            )
        };

        let group_id = *population
            .group_ids
            .get(node_index)
            .ok_or_else(node_index_error)?;
        let node_type_id = *population
            .type_ids
            .get(node_index)
            .ok_or_else(node_index_error)?;
        let group_index = *population
            .group_indices
            .get(node_index)
            .ok_or_else(node_index_error)?;
        let mut incoming_edges = Vec::new();
        for edge_list in &self.edge_lists {
            for edge in &edge_list.populations {
                if edge.target_pop == population.name {
                    if let Some(edge_index) = edge.target_ids.iter().position(|it| it == node_id) {
                        let type_id =
                            *edge.type_ids.get(edge_index).ok_or_else(node_index_error)?;
                        let src_id = edge
                            .source_ids
                            .get(edge_index)
                            .ok_or_else(node_index_error)?;
                        let src_pop = &edge.source_pop;
                        let src_idx = self
                            .population_ids
                            .get(src_pop)
                            .ok_or_else(|| anyhow!("Unknown population {src_pop}"))?;
                        let src_gid = *self.population_to_gid.get(&(*src_idx, *src_id)).unwrap();
                        let group_id = *edge
                            .group_ids
                            .get(edge_index)
                            .ok_or_else(node_index_error)?;
                        let index = *edge
                            .group_indices
                            .get(edge_index)
                            .ok_or_else(node_index_error)?;
                        let ty = edge_list
                            .types
                            .iter()
                            .find(|ty| ty.type_id == type_id)
                            .ok_or_else(|| {
                                anyhow!(
                                    "Couldn't find edge type {type_id} in population {}",
                                    edge.name
                                )
                            })?;
                        let delay = if let Some(d) = ty.attributes.get("delay") {
                            if let Attribute::Float(d) = d {
                                *d
                            } else {
                                bail!("Edge type {type_id} in population {} has non-float delay", edge.name);
                            }
                        } else {
                            bail!("Edge type {type_id} in population {} has no delay", edge.name);
                        };
                        let weight = if let Some(d) = ty.attributes.get("syn_weight") {
                            if let Attribute::Float(d) = d {
                                *d
                            } else {
                                bail!("Edge type {type_id} in population {} has non-float weight", edge.name);
                            }
                        } else {
                            bail!("Edge type {type_id} in population {} has no weight", edge.name);
                        };

                        let mech = if let Some(s) = ty.attributes.get("model_template") {
                            if let Attribute::String(s) = s {
                                s.to_string()
                            } else {
                                bail!("Edge type {type_id} in population {} has non-string model", edge.name);
                            }
                        } else {
                            String::from("NULL") // Yuck
                        };

                        let mut dynamics = Map::new();
                        if let Some(s) = ty.attributes.get("dynamics_params") {
                            if let Attribute::String(s) = s {
                                if let Some(d) = edge_list.dynamics.get(s) {
                                    dynamics = d.clone();
                                } else {
                                    bail!("Edge type {type_id} in population {} has unknown dynamics {s}", edge.name);
                                }
                            } else {
                                bail!("Edge type {type_id} in population {} has non-string params", edge.name);
                            }
                        }

                        incoming_edges.push(Edge {
                            pop: edge.name.to_string(),
                            group_id,
                            index,
                            type_id,
                            src_gid,
                            mech,
                            delay,
                            weight,
                            dynamics,
                        });
                    }
                }
            }
        }
        let node_type = node_list
            .types
            .iter()
            .find(|ty| ty.type_id == node_type_id)
            .ok_or_else(|| {
                anyhow!(
                    "Couldn't find node type {node_type_id} in population {}",
                    population.name
                )
            })?
            .clone();
        let group = population
            .groups
            .iter()
            .find(|g| g.id == group_id)
            .ok_or_else(|| {
                anyhow!(
                    "Couldn't find group id {group_id} node population {}",
                    population.name
                )
            })?;
        let dynamics = group
            .dynamics
            .iter()
            .map(|(k, vs)| (k.to_string(), vs[group_index]))
            .collect();
        let custom = group
            .custom
            .iter()
            .map(|(k, vs)| (k.to_string(), vs[group_index]))
            .collect();
        Ok(Node {
            gid,
            pop: population.name.clone(),
            group_id,
            node_id: node_index as u64,
            group_index,
            node_type_id,
            node_type,
            incoming_edges,
            dynamics,
            custom,
        })
    }
}

/// Resources to store in the output.
#[derive(Debug)]
struct Bundle {
    /// Python source for the recipe.
    recipe: String,
    /// A list of morphologies to copy.
    mrfs: Vec<String>,
    /// A list of fits to generate.
    fits: Vec<String>,
}

fn gen_recipe_py(sim: &Simulation) -> Result<Bundle> {
    let mut gid_to_mid = Map::new();
    let mut gid_to_cid = Map::new();
    let mut gid_to_kid = Vec::new();
    let mut mid_to_mrf = Vec::new();
    let mut mrf_to_mid = Map::new();
    let mut gid_to_inc = Vec::new();

    let mut fits = Vec::new();

    for gid in 0..sim.size {
        let node = sim.reify_node(gid)?;
        let mut inc = Vec::new();
        for edge in node.incoming_edges {
            inc.push((edge.src_gid, edge.mech, edge.dynamics, edge.weight, edge.delay));
        }
        gid_to_inc.push(inc);
        match &node.node_type.model_type {
            ModelType::Biophysical { model_template, attributes } => {
                match model_template.as_ref() {
                    "ctdb:Biophys1.hoc" => {
                        if let Some(Attribute::String(mrf)) = attributes.get("morphology") {
                            if !mrf_to_mid.contains_key(mrf) {
                                let mid = mid_to_mrf.len();
                                mid_to_mrf.push(mrf.to_string());
                                mrf_to_mid.insert(mrf.to_string(), mid);
                            }
                            let mid = mrf_to_mid[mrf];
                            gid_to_mid.insert(gid, mid);
                        } else {
                            bail!("GID {gid} is a biophysical cell, but has no morphology.");
                        }
                        gid_to_kid.push(Some(0));
                        if let Some(Attribute::String(fit)) = attributes.get("dynamics_params") {
                            fits.push(fit.to_string());
                            gid_to_cid.insert(gid, fit.split_once('.').unwrap().0.to_string());
                        } else {
                            bail!("GID {gid} is a biophysical cell, but has no dynamics_params.");
                        }
                    }
                    t => bail!("Unknown model template <{t}> for gid {gid}"),
                }
            }
            ModelType::Virtual { attributes } => {
                gid_to_kid.push(Some(2));
            }
            ModelType::Point { attributes, model_template } => {
                gid_to_kid.push(Some(1));
            }
            _ => gid_to_kid.push(None),
        }
    }

    let mut py_gid_to_mid = String::from("{");
    for (gid, mid) in gid_to_mid.iter() {
        py_gid_to_mid.push_str(&format!("{gid}: {mid}, "))
    }
    py_gid_to_mid.push_str("}");

    let mut py_mid_to_mrf = String::from("[");
    for mrf in mid_to_mrf.iter() {
        py_mid_to_mrf.push_str(&format!("'{mrf}', "))
    }
    py_mid_to_mrf.push_str("]");

    let mut py_gid_to_kid = String::from("[");
    for kid in gid_to_kid.iter() {
        if let Some(kid) = kid {
            py_gid_to_kid.push_str(&format!("{kid}, "));
        } else {
            py_gid_to_kid.push_str("None, ");
        }
    }
    py_gid_to_kid.push_str("]");

    let mut py_gid_to_cid = String::from("{");
    for (gid, cid) in gid_to_cid.iter() {
        py_gid_to_cid.push_str(&format!("{gid}: '{cid}', "))
    }
    py_gid_to_cid.push_str("}");

    let mut py_gid_to_inc = String::from("[");
    for inc in gid_to_inc.iter() {
        py_gid_to_inc.push_str("[");
        for (src, syn, ps, w, d) in inc {
            let mut py_ps = String::from("{");
            for (k, v) in ps.iter() {
                py_ps.push_str(&format!("'{k}': {v}, ", k = k, v = v));
            }
            py_ps.push_str("}");
            py_gid_to_inc.push_str(&format!("({src}, \"{syn}\", {py_ps}, {w}, {d}), "));
        }
        py_gid_to_inc.push_str("], ");
    }
    py_gid_to_inc.push_str("]");

    let recipe = format!("
class recipe(A.recipe):

    def __init__(self):
        A.recipe.__init__(self)
        self.cells = {}
        # gid -> morphology id
        self.gid_to_mid = {}
        # morphology id -> morphology resource file
        self.mid_to_mrf = {}
        # gid -> cell kind
        self.gid_to_kid = {}
        # gid -> cell description
        self.gid_to_cid = {}
        # gid -> incoming connections
        self.gid_to_inc = {}
        # spike threshold
        self.threshold = -15

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
        return self.cells

    def connections_on(self, gid):
        kind = self.gid_to_kid[gid]
        if kind == 0:
            return [A.connection((src, 'src'), 'tgt'+str(ix), w, d * U.ms)
                    for ix, (src, _, _, w, d)
                    in enumerate(self.gid_to_inc[gid])]
        else:
            return [A.connection((src, 'src'), 'tgt', w, d * U.ms)
                    for (src, _, _, w, d)
                    in self.gid_to_inc[gid]]

    def global_properties(self, kind):
        if kind == A.cell_kind.cable:
            return self.cable_props
        raise RuntimeError('Unexpected cell kind')

    def cell_description(self, gid):
        kind = self.gid_to_kid[gid]
        if kind == 0:
            mid = self.gid_to_mid[gid]
            mrf = load_morphology(here / 'mrf' / self.mid_to_mrf[mid])
            dec = A.load_component(here / 'acc' / (self.gid_to_cid[gid] + '.acc')).component
            dec.place('(location 0 0.5)', A.threshold_detector(self.threshold * U.mV), 'src')
            for ix, (_, syn, params, _, _) in enumerate(self.gid_to_inc[gid]):
                #NB. fix parameters?! dec.place('(location 0 0.5)', A.synapse(syn, **params), 'tgt'+str(ix))
                dec.place('(location 0 0.5)', A.synapse(syn), 'tgt'+str(ix))
            lbl = A.label_dict()
            lbl = lbl.add_swc_tags()
            return A.cable_cell(mrf, dec, lbl)
        elif kind == 1:
            return A.lif_cell('src', 'tgt')
        elif kind == 2:
            return A.spike_source_cell('src', A.explicit_schedule([]))
        else:
            raise RuntimeError('Unknown cell kind')

", sim.size, py_gid_to_mid, py_mid_to_mrf, py_gid_to_kid, py_gid_to_cid, py_gid_to_inc);

    Ok(Bundle { recipe, mrfs: mid_to_mrf, fits })
}

fn find_component(file: &str, components: &Map<String, String>) -> Result<std::path::PathBuf> {
    for pth in components.values() {
        let mut src = std::path::PathBuf::from_str(pth)?;
        src.push(file);
        if src.exists() {
            return Ok(src);
        }
        src.pop();
    }
    bail!("Couldn't find required resource {file}");
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Fit { from } => {
            let ifn = std::path::PathBuf::from_str(&from)
                .map_err(anyhow::Error::from)
                .with_context(|| format!("Resolving input file {from}"))?;
            let raw = Fit::from_file(&ifn)
                .with_context(|| format!("Parsing fit {from}"))?;
            eprintln!("Raw Fit: {raw:#?}");
            let dec = raw.decor()?;
            eprintln!("Mechanisms: {:#?}", dec);
            let acc = dec.to_acc()?;
            eprintln!("ACC: {}", acc);
            Ok(())
        },
        Cmd::Build { from, to } => {
            let raw = raw::Simulation::from_file(&from)
                .with_context(|| format!("Parsing simulation {from}"))?;
            eprintln!("Raw Simulation: {raw:#?}");
            let sim = Simulation::new(&raw)?;
            let out = gen_recipe_py(&sim).with_context(|| "Generating Python code")?;

            // Create all required directories
            let mut to = std::path::PathBuf::from_str(&to)
                .map_err(anyhow::Error::from)
                .with_context(|| format!("Resolving output dir {to}"))?;
            std::fs::create_dir_all(&to).with_context(|| format!("Creating output dir {to:?}"))?;

            to.push("mrf");
            std::fs::create_dir_all(&to).with_context(|| format!("Creating output dir {to:?}"))?;

            for mrf in &out.mrfs {
                let src = find_component(mrf, &raw.components)?;
                to.push(mrf);
                std::fs::copy(&src, &to).with_context(|| format!("Copying {src:?} to {to:?}"))?;
                to.pop();
            }
            to.pop();

            to.push("acc");
            std::fs::create_dir_all(&to).with_context(|| format!("Creating output dir {to:?}"))?;

            for fit in &out.fits {
                to.push(fit);
                to.set_extension("acc");
                let src = find_component(&fit, &raw.components)?;
                let inp = Fit::from_file(&src)?.decor()?.to_acc()?;
                std::fs::write(&to, inp).with_context(|| format!("Writing {to:?}"))?;
                to.pop();
            }
            to.pop();

            to.push("dat");
            std::fs::create_dir_all(&to).with_context(|| format!("Creating output dir {to:?}"))?;
            to.pop();

            to.push("out");
            std::fs::create_dir_all(&to).with_context(|| format!("Creating output dir {to:?}"))?;
            to.pop();

            to.push("main.py");
            std::fs::write(&to,
                           format!("import arbor as A
from arbor import units as U

from pathlib import Path

here = Path(__file__).parent

def load_morphology(path):
    sfx = path.suffix
    if sfx == '.swc':
        try:
            return A.load_swc_arbor(path)
        except:
            pass
        try:
            return A.load_swc_neuron(path)
        except:
            raise RuntimeError(f\"Could load {{path}} neither as NEURON nor Arbor flavour.\")
    elif sfx == '.asc':
        return A.load_asc(path).morphology
    elif sfx == '.nml':
        nml = A.load_nml(path)
        if len(nml.morphology_ids()) == 1:
            return nml.morphology(nml.morphology_ids()[0]).morphology
        else:
            raise RuntimeError(f\"NML file {{path}} contains multiple morphologies.\")
    else:
        raise RuntimeError(f\"Unknown morphology file type {{sfx}}\")

{}
rec = recipe()
sim = A.simulation(rec)
sim.run({}*U.ms, {}*U.ms)
", out.recipe, sim.tfinal, sim.dt)).with_context(|| format!("Creating simulation file {to:?}"))?;
            to.pop();

            Ok(())
        }
    }
}
