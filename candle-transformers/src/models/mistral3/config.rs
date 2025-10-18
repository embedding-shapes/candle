use candle_nn::Activation;
use serde::Deserialize;

// Constants/configurable defaults placed below imports as per AGENTS.md
fn default_image_token_index() -> usize {
    10
}

fn default_projector_hidden_act() -> Activation {
    Activation::Gelu
}

fn default_multimodal_projector_bias() -> bool {
    false
}

fn default_spatial_merge_size() -> usize {
    2
}

fn default_vision_feature_layer() -> VisionFeatureLayer {
    VisionFeatureLayer::Single(-1)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum VisionFeatureLayer {
    Single(i64),
    List(Vec<i64>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct Mistral3Config {
    pub model_type: String,

    #[serde(default = "default_image_token_index")]
    pub image_token_index: usize,

    #[serde(default = "default_projector_hidden_act")]
    pub projector_hidden_act: Activation,

    #[serde(default = "default_multimodal_projector_bias")]
    pub multimodal_projector_bias: bool,

    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,

    #[serde(default = "default_vision_feature_layer")]
    pub vision_feature_layer: VisionFeatureLayer,

    pub text_config: TextSubConfig,
    pub vision_config: VisionSubConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextSubConfig {
    #[serde(flatten)]
    pub inner: crate::models::mistral::Config,
    pub model_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionSubConfig {
    #[serde(flatten)]
    pub inner: crate::models::pixtral::vision_model::Config,
    pub model_type: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn hub_root() -> PathBuf {
        if let Ok(p) = std::env::var("HF_HOME") {
            return PathBuf::from(p);
        }
        if let Ok(p) = std::env::var("HF_HUB_CACHE") {
            return PathBuf::from(p);
        }
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/"));
        home.join(".cache").join("huggingface").join("hub")
    }

    fn title_case_repo(repo: &str) -> String {
        repo.split('-')
            .map(|s| {
                let mut chars = s.chars();
                match chars.next() {
                    None => String::new(),
                    Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
                }
            })
            .collect::<Vec<_>>()
            .join("-")
    }

    fn is_complete_snapshot(p: &Path) -> bool {
        let idx = p.join("model.safetensors.index.json");
        if !idx.is_file() {
            return false;
        }
        if let Ok(read_dir) = fs::read_dir(p) {
            for entry in read_dir.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("model-") && name.ends_with(".safetensors") {
                    if entry.path().is_file() {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn find_local_snapshot(repo_id: &str) -> Option<PathBuf> {
        let (org, repo) = repo_id.split_once('/')?;
        let hub = hub_root();
        let candidates = [
            hub.join(format!("models--{}--{}", org, repo)).join("snapshots"),
            hub.join(format!("models--{}--{}", org, title_case_repo(repo))).join("snapshots"),
        ];
        let mut snaps: Vec<PathBuf> = vec![];
        for root in candidates.iter() {
            if root.is_dir() {
                if let Ok(rd) = fs::read_dir(root) {
                    for e in rd.flatten() {
                        if e.path().is_dir() {
                            snaps.push(e.path());
                        }
                    }
                }
            }
        }
        if snaps.is_empty() {
            return None;
        }
        snaps.sort_by_key(|p| fs::metadata(p).and_then(|m| m.modified()).ok());
        snaps.reverse();
        for s in &snaps {
            if is_complete_snapshot(s) {
                return Some(s.clone());
            }
        }
        Some(snaps.remove(0))
    }

    #[test]
    fn deserialize_magistral_config() -> Result<(), Box<dyn std::error::Error>> {
        let repo = "mistralai/magistral-small-2509";
        let Some(snapshot) = find_local_snapshot(repo) else {
            eprintln!(
                "no local HF snapshot found for {} under {:?}",
                repo,
                hub_root()
            );
            // Skip the test when no local snapshot is present.
            return Ok(());
        };
        let cfg_path = snapshot.join("config.json");
        eprintln!("using snapshot: {:?}", snapshot);
        eprintln!("loading config: {:?}", cfg_path);
        let txt = fs::read_to_string(&cfg_path)?;
        let cfg: Mistral3Config = serde_json::from_str(&txt)?;

        assert_eq!(cfg.model_type, "mistral3");
        assert_eq!(cfg.image_token_index, 10);

        assert_eq!(cfg.vision_config.model_type.as_deref(), Some("pixtral"));
        assert_eq!(cfg.text_config.model_type.as_deref(), Some("mistral"));

        // Sanity: ensure we captured inner configs too
        // These are the crucial fields we expect from the snapshot
        assert_eq!(cfg.text_config.inner.num_attention_heads, 32);
        assert_eq!(cfg.vision_config.inner.num_attention_heads, 16);

        Ok(())
    }
}
