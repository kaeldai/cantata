use clap::{self, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sonata::{
    err::{Context, Result},
    raw, Map,
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
    Build { from: String, to: Option<String> },
}

#[derive(Debug, Serialize, Deserialize)]
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
#[derive(Debug, Serialize, Deserialize)]
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
/// - additional columns may be required if defined by the population or in
///   the associated node types CSV.
#[derive(Debug, Serialize, Deserialize)]
struct NodeType {
    #[serde(rename = "node_type_id")]
    type_id: u64,
    #[serde(rename = "pop_name")]
    population: Option<String>,
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
                attributes.retain(|_, v| !matches!(v, Attribute::String(s) if s == "NULL"));
            }
        }
    }
}

#[derive(Debug)]
struct NodeGroup {
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
///         * /<group_id>                Group
///             * /dynamics_params       Group
///                 * /<param>           Dataset {M_nodes}
///             * /<custom_attribute>    Dataset {M_nodes}
///
/// Notes:
/// * For each unique entry in node_group_id we expect one <group_id> group
///   under the population
#[derive(Debug)]
struct Population {
    size: usize,
    type_ids: Vec<u64>,
    node_ids: Vec<u64>,
    group_ids: Vec<u64>,
    group_indices: Vec<u64>,
    groups: Vec<NodeGroup>,
}

#[derive(Debug)]
struct NodeList {
    types: Vec<NodeType>,
    populations: Vec<Population>,
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

        let mut pops = Vec::new();
        for population in &populations {
            let name = population.name();
            let type_ids = population
                .dataset("node_type_id")
                .with_context(|| format!("Extracting node_type_id from population {name}"))?
                .read_1d::<u64>()?
                .to_vec();
            let size = type_ids.len();
            let node_ids = population
                .dataset("node_id")
                .with_context(|| format!("Extracting node_id from population {name}"))?
                .read_1d::<u64>()?
                .to_vec();
            if size != node_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #node_ids"
                )
            }
            let group_ids = population
                .dataset("node_group_id")
                .with_context(|| format!("Extracting group index from population {name}"))?
                .read_1d::<u64>()?
                .to_vec();
            if size != group_ids.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #group_id"
                )
            }
            let group_indices = population
                .dataset("node_group_index")
                .with_context(|| format!("Extracting group index from population {name}"))?
                .read_1d::<u64>()?
                .to_vec();
            if size != group_indices.len() {
                anyhow::bail!(
                    "Population {name} in {path:?} has mismatched #type_ids ./. #group_index"
                )
            }
            let mut groups = Vec::new();
            let mut group = 0;
            loop {
                if let Ok(group) = population.group(&format!("{group}")) {
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

                    groups.push(NodeGroup { dynamics, custom });
                } else {
                    break;
                }
                group += 1;
            }
            pops.push(Population {
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
        })
    }
}

/// types are defined in a CSV file of named columns; separator is a single space.
/// - edge_type_id; required
/// - population; required; handles populations defining the same edge_type_id.
/// - any number of additional columns may freely be added.
#[derive(Debug, Serialize, Deserialize)]
struct EdgeType {
    #[serde(rename = "edge_type_id")]
    type_id: u64,
    #[serde(rename = "pop_name")]
    population: String,
    #[serde(flatten)]
    attributes: Map<String, serde_json::value::Value>,
}

#[derive(Debug)]
pub struct Simulation {
    node_lists: Vec<NodeList>,
}

impl Simulation {
    fn new(sim: &raw::Simulation) -> Result<Self> {
        let mut node_lists = Vec::new();
        for nodes in sim.network.nodes.iter() {
            node_lists.push(NodeList::new(nodes)?);
        }
        Ok(Self { node_lists })
    }
}

fn reify_edges(edges: &raw::Edges) -> Result<Vec<EdgeType>> {
    let path = &edges.types;
    let path = std::path::PathBuf::from_str(&edges.types)
        .map_err(anyhow::Error::from)
        .and_then(|p| p.canonicalize().map_err(anyhow::Error::from))
        .with_context(|| format!("Resolving node types {path}"))?;
    let rd = File::open(&path).with_context(|| format!("Opening {path:?}"))?;
    let ty = csv::ReaderBuilder::new()
        .delimiter(b' ')
        .from_reader(rd)
        .deserialize()
        .map(|it| it.map_err(anyhow::Error::from))
        .collect::<Result<Vec<EdgeType>>>()
        .with_context(|| format!("Parsing edge types {path:?}"));
    ty
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Build { from, .. } => {
            let sim = raw::Simulation::from_file(&from)
                .with_context(|| format!("Parsing simulation {from}"))?;
            let sim = Simulation::new(&sim);
            eprintln!("Simulation: {sim:#?}");
            Ok(())
        }
    }
}
