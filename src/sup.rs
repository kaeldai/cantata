use crate::{err::Result, Map};
use anyhow::bail;
use std::str::FromStr;

pub fn find_component(file: &str, components: &Map<String, String>) -> Result<std::path::PathBuf> {
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
