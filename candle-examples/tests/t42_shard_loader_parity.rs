use anyhow::{Context, Result};
use candle::safetensors::{MmapedFile, MmapedSafetensors};
use candle::Device;
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::PathBuf;

const SNAPSHOT_DIR: &str = "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";
const INDEX_FILE: &str = "model.safetensors.index.json";
const FIRST_N_BYTES: usize = 64 * 1024; // 64 KiB

fn expand_tilde(p: &str) -> std::io::Result<PathBuf> {
    if let Some(rest) = p.strip_prefix("~/") {
        Ok(std::env::var("HOME").map(PathBuf::from).unwrap().join(rest))
    } else {
        Ok(PathBuf::from(p))
    }
}

fn sha256_bytes(buf: &[u8]) -> Result<String> {
    // Prefer sha256sum if available; else fall back to shasum -a 256.
    // We avoid adding new Rust deps per project principles.
    let try_sha256sum = || -> Result<String> {
        let mut child = std::process::Command::new("sha256sum")
            .arg("-")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .context("spawn sha256sum")?;
        {
            let stdin = child.stdin.as_mut().unwrap();
            stdin.write_all(buf).context("write stdin sha256sum")?;
        }
        let out = child.wait_with_output().context("run sha256sum")?;
        if !out.status.success() {
            anyhow::bail!("sha256sum failed: {}", out.status);
        }
        let s = String::from_utf8_lossy(&out.stdout);
        Ok(s.split_whitespace().next().unwrap_or("").to_string())
    };
    match try_sha256sum() {
        Ok(s) if !s.is_empty() => Ok(s),
        _ => {
            let mut child = std::process::Command::new("shasum")
                .args(["-a", "256", "-"])
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .spawn()
                .context("spawn shasum")?;
            {
                let stdin = child.stdin.as_mut().unwrap();
                stdin.write_all(buf).context("write stdin shasum")?;
            }
            let out = child.wait_with_output().context("run shasum")?;
            if !out.status.success() {
                anyhow::bail!("shasum failed: {}", out.status);
            }
            let s = String::from_utf8_lossy(&out.stdout);
            Ok(s.split_whitespace().next().unwrap_or("").to_string())
        }
    }
}

#[test]
fn t42_sharded_safetensors_loader_mapping_parity() -> Result<()> {
    // Ensure CUDA is visible as requested (the test itself reads CPU bytes from memmap).
    let dev = Device::cuda_if_available(0)?;
    eprintln!("device: {:?}", dev);

    let snap = expand_tilde(SNAPSHOT_DIR)?;
    let idx_path = snap.join(INDEX_FILE);
    if !idx_path.exists() {
        eprintln!("index not found at {} — skipping", idx_path.display());
        return Ok(());
    }

    // 1) Parse model.safetensors.index.json → weight_map
    #[derive(serde::Deserialize)]
    struct Idx {
        weight_map: BTreeMap<String, String>,
    }
    let idx_bytes =
        std::fs::read(&idx_path).with_context(|| format!("read {}", idx_path.display()))?;
    let idx: Idx =
        serde_json::from_slice(&idx_bytes).context("invalid model.safetensors.index.json")?;

    // 2) Build loader union and our own name→file index map by scanning headers.
    let files = candle_examples::hub_load_local_safetensors(&snap, INDEX_FILE)?;
    assert!(
        !files.is_empty(),
        "no safetensors files resolved from index"
    );
    let union = unsafe { MmapedSafetensors::multi(&files)? };

    let mut name_to_file_idx: HashMap<String, usize> = HashMap::new();
    let mut file_idx_to_name: HashMap<usize, String> = HashMap::new();
    for (i, p) in files.iter().enumerate() {
        file_idx_to_name.insert(i, p.file_name().unwrap().to_string_lossy().into_owned());
        let mf = unsafe { MmapedFile::new(p)? };
        let st = mf.deserialize()?;
        for n in st.names() {
            // last file wins, mirroring Candle routing
            name_to_file_idx.insert(n.to_string(), i);
        }
    }

    // 3) Iterate all tensors listed in the weight_map
    for name in idx.weight_map.keys() {
        let name = name.as_str();
        let expected_file = idx
            .weight_map
            .get(name)
            .unwrap_or_else(|| panic!("missing '{name}' in weight_map"));

        // Loader-resolved file index and path
        let idx_loader = *name_to_file_idx
            .get(name)
            .unwrap_or_else(|| panic!("name '{name}' not present in any shard headers"));
        let loader_file = file_idx_to_name.get(&idx_loader).unwrap().clone();

        // 4a) Assert loader’s shard filename equals weight_map[name]
        assert_eq!(
            &loader_file, expected_file,
            "loader shard mismatch for {name}: loader='{}' expected='{}'",
            loader_file, expected_file
        );

        // 4b) Compute SHA256 over first 64KiB of raw bytes from that shard
        let shard_path = snap.join(expected_file);
        let mf = unsafe { MmapedFile::new(&shard_path)? };
        let st = mf.deserialize()?;
        let tv = st.tensor(name)?;
        let raw = tv.data();
        let take_n = std::cmp::min(FIRST_N_BYTES, raw.len());
        let expected_hash = sha256_bytes(&raw[..take_n])?;

        // 4c) Compute SHA256 over bytes from Candle loader (pre-copy view)
        let tv2 = union.get(name)?;
        let raw2 = tv2.data();
        let take_n2 = std::cmp::min(FIRST_N_BYTES, raw2.len());
        let actual_hash = sha256_bytes(&raw2[..take_n2])?;

        // Log dtype, shape, sizes
        eprintln!(
            "name='{}' file='{}' dtype={:?} shape={:?} nbytes={} first={} actual={} expected={}",
            name,
            loader_file,
            tv.dtype(),
            tv.shape(),
            raw.len(),
            take_n,
            actual_hash,
            expected_hash
        );

        assert_eq!(
            actual_hash, expected_hash,
            "byte-slice sha256 mismatch for {}",
            name
        );
    }

    Ok(())
}
