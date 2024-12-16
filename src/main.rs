use cantata::{
    err::{Context, Result},
    fit::Fit,
    gen::Bundle,
    nml, raw,
    sim::Simulation,
    sup::find_component,
};
use clap::{self, Parser, Subcommand, ValueEnum};
use std::str::FromStr;

#[derive(Parser)]
#[clap(name = "sonata")]
#[clap(version = "0.1.0-dev", author = "t.hater@fz-juelich.de")]
struct Cli {
    #[clap(subcommand)]
    cmd: Cmd,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Hash, PartialEq, PartialOrd, Eq, Ord, Clone, Copy, ValueEnum)]
enum Format {
    CBOR,
    JSON,
    Pickle,
}

#[derive(Subcommand)]
enum Cmd {
    Build {
        from: String,
        to: String,
        #[arg(short, long)]
        formats: Vec<Format>,

        #[arg(short, long, default_value="0.9-dev")]
        arbor_version: String,
    },
    Run {
        from: String,
        #[arg(short, long)]
        to: Option<String>,

        #[arg(short, long, default_value="0.9-dev")]
        arbor_version: String,
    },
}

fn build(from: &str, to: &str, formats: &[Format], ver_str: &str) -> Result<()> {
    let raw =
        raw::Simulation::from_file(from).with_context(|| format!("Parsing simulation {from}"))?;
    let sim = Simulation::new(&raw).with_context(|| format!("Extracting simulation {from}"))?;
    let mut out = Bundle::new(&sim).with_context(|| "Generating Python code")?;

    // Create all required directories
    let mut to = std::path::PathBuf::from_str(to)
        .map_err(anyhow::Error::from)
        .with_context(|| format!("Resolving output dir {to}"))?;
    std::fs::create_dir_all(&to).with_context(|| format!("Creating output dir {to:?}"))?;

    to.push("mrf");
    std::fs::create_dir_all(&to).with_context(|| format!("Creating output dir {to:?}"))?;

    for mrf in &out.morphology {
        let src = find_component(mrf, &raw.components)
            .with_context(|| format!("Searching morphology {mrf:?}"))?;
        to.push(mrf);
        std::fs::copy(&src, &to).with_context(|| format!("Copying {src:?} to {to:?}"))?;
        to.pop();
    }
    to.pop();

    to.push("acc");
    std::fs::create_dir_all(&to).with_context(|| format!("Creating output dir {to:?}"))?;

    for fit in out.decoration.iter_mut() {
        to.push(fit as &str);
        to.set_extension("acc");
        let src = find_component(fit, &raw.components)
            .with_context(|| format!("Searching raw fit {fit:?}"))?;
        match src.extension().and_then(|s| s.to_str()) {
            Some("json") => {
                let inp = Fit::from_file(&src)
                    .with_context(|| format!("Extracting fit {src:?}"))?
                    .decor()
                    .with_context(|| format!("Building decor for fit {src:?}"))?
                    .to_acc(&ver_str)
                    .with_context(|| format!("Converting fit {src:?} to acc"))?;
                std::fs::write(&to, inp).with_context(|| format!("Writing {to:?}"))?;
            }
            Some("nml") => {
                let data = find_component(fit, &sim.components)
                    .with_context(|| format!("Searching raw fit {fit:?}"))?;
                std::fs::write(&to, nml::mk_acc(&data)?)
                    .with_context(|| format!("Writing {to:?}"))?;
            }
            Some(e) => anyhow::bail!("Unknown fit type {e}"),
            None => anyhow::bail!("Unspecified fit type."),
        }
        to.pop();
        *fit = format!("{}.acc", fit.rsplit_once('.').expect(".json?").0);
    }
    to.pop();

    to.push("dat");
    std::fs::create_dir_all(&to).with_context(|| format!("Creating output dir {to:?}"))?;
    if formats.contains(&Format::JSON) {
        to.push("sim.json");
        let writer = std::fs::File::create(&to)?;
        serde_json::to_writer_pretty(writer, &out)?;
        to.pop();
    }
    if formats.is_empty() || formats.contains(&Format::CBOR) {
        to.push("sim.cbor");
        let writer = std::fs::File::create(&to)?;
        ciborium::into_writer(&out, writer)?;
        to.pop();
    }
    if formats.contains(&Format::Pickle) {
        to.push("sim.pcl");
        let mut writer = std::fs::File::create(&to)?;
        serde_pickle::to_writer(&mut writer, &out, Default::default())?;
        to.pop();
    }

    to.pop();

    to.push("out");
    std::fs::create_dir_all(&to).with_context(|| format!("Creating output dir {to:?}"))?;
    to.pop();

    to.push("main.py");
    std::fs::write(&to, include_str!("../data/main.py"))
        .with_context(|| format!("Copying simulation file {to:?}"))?;
    to.pop();

    Ok(())
}

fn main() -> Result<()> {
    let mut cli = Cli::parse();
    match cli.cmd {
        Cmd::Build {
            from,
            to,
            ref mut formats,
            arbor_version,
        } => {
            if formats.is_empty() {
                formats.push(Format::CBOR);
            }
            // print!("{arbor_version:?}");
            build(&from, &to, formats, &arbor_version)
        }
        Cmd::Run { 
            from, 
            to,
            arbor_version,
         } => {
            let to = if let Some(to) = to {
                to.to_string()
            } else {
                let mut to = std::path::PathBuf::from_str(&from)?;
                to.set_extension("sim");
                to.file_name().and_then(|s| s.to_str()).unwrap().to_string()
            };
            build(&from, &to, &[Format::CBOR], &arbor_version)?;
            let _ = std::process::Command::new("python3")
                .current_dir(to)
                .arg("main.py")
                .status()?;
            Ok(())
        }
    }
}
