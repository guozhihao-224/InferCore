use crate::tokenizer::error::TokenizerSnafu;
use snafu::ResultExt;
use tokenizers::{Tokenizer, models::bpe::BPE};

use super::error::Result;

#[derive(Debug, Clone)]
pub struct QwenTokenizer {
    tokenizer: Tokenizer,
}

impl QwenTokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).context(TokenizerSnafu)?;
        Ok(Self { tokenizer })
    }

    pub fn from_vocab_and_merges(vocab_path: &str, merges_path: &str) -> Result<Self> {
        let bpe = BPE::from_file(vocab_path, merges_path)
            .build()
            .context(TokenizerSnafu)?;
        let tokenizer = Tokenizer::new(bpe);
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        println!("[encode] input: {text}");
        let encoding = self.tokenizer.encode(text, true).context(TokenizerSnafu)?;
        let ids = encoding.get_ids().to_vec();
        println!("[encode] token ids: {ids:?}");
        Ok(ids)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        println!("[decode] input token ids: {tokens:?}");
        let text = self
            .tokenizer
            .decode(tokens, true)
            .context(TokenizerSnafu)?;
        println!("[decode] output: {text}");
        Ok(text)
    }
}
