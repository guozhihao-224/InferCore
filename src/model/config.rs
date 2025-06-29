use std::fs;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct QwenConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f32,
    // 其他字段可选
}

impl QwenConfig {
    pub fn from_file(path: &str) -> Result<Self, String> {
        let s = fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&s).map_err(|e| e.to_string())
    }
}

#[derive(Debug, Deserialize)]
pub struct GenerationConfig {
    pub bos_token_id: u32,
    pub pad_token_id: u32,
    pub do_sample: bool,
    pub eos_token_id: Vec<u32>,
    pub repetition_penalty: f32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub transformers_version: Option<String>,
}

impl GenerationConfig {
    pub fn from_file(path: &str) -> Result<Self, String> {
        let s = fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&s).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_qwen_config_from_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_config.json");
        let mut file = File::create(&file_path).unwrap();
        let json = r#"{
            "vocab_size": 100,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "rms_norm_eps": 1e-6
        }"#;
        file.write_all(json.as_bytes()).unwrap();
        let config = QwenConfig::from_file(file_path.to_str().unwrap()).unwrap();
        assert_eq!(config.vocab_size, 100);
        assert_eq!(config.hidden_size, 32);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.intermediate_size, 64);
        assert_eq!(config.num_hidden_layers, 2);
        assert!((config.rms_norm_eps - 1e-6).abs() < 1e-9);
    }
}
