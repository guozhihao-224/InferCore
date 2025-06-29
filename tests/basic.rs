// use std::fs;

// use qwen3_infer::model::transformer::QwenTransformer;
// use qwen3_infer::tensor::device::Device;
// use qwen3_infer::tokenizer::tokenizer::QwenTokenizer;
// use safetensors::SafeTensors;

// #[test]
// fn test_tokenizer() {
//     let tokenizer =
//         QwenTokenizer::from_vocab_and_merges("Qwen2-0.5B/vocab.json", "Qwen2-0.5B/merges.txt")
//             .unwrap();

//     let tokens = tokenizer.encode("what's your name").unwrap();
//     println!("{:?}", tokens);
//     let text = tokenizer.decode(&tokens).unwrap();
//     assert_eq!(text, "what 's your name");
// }

// #[test]
// fn test_tokenizer_special_tokens() {
//     let tokenizer =
//         QwenTokenizer::from_file("Qwen2.5-0.5B-instruct/tokenizer.json")
//             .unwrap();

//     // 测试 BOS/EOS
//     let bos_id = 151643;
//     let bos_text = tokenizer.decode(&[bos_id]).unwrap();
//     println!("BOS decode: {}", bos_text);

//     // 测试中英文混合
//     let tokens = tokenizer.encode("你好，Qwen!").unwrap();
//     println!("中英文分词: {:?}", tokens);
//     let text = tokenizer.decode(&tokens).unwrap();
//     println!("中英文还原: {}", text);
// }

// #[test]
// fn test_load_real_safetensors_weights() {
//     // 加载 config
//     let config =
//         qwen3_infer::model::config::QwenConfig::from_file("Qwen2-0.5B/config.json").unwrap();
//     // 初始化模型
//     let mut model = QwenTransformer::new(
//         config.vocab_size,
//         config.hidden_size,
//         config.num_attention_heads,
//         config.num_key_value_heads,
//         config.intermediate_size,
//         config.num_hidden_layers,
//         config.rms_norm_eps,
//         Device::Cpu,
//     );
//     // 加载权重
//     model
//         .load_weights_safetensors("Qwen2-0.5B/model.safetensors")
//         .unwrap();
//     // 检查 embedding、lm_head、block0.attn.q_proj 前10项
//     let emb = model.embedding.to_vec();
//     let lm_head = model.lm_head.to_vec();
//     let q_proj = model.blocks[0].attn.q_proj.to_vec();
//     println!("embedding (前10): {:?}", &emb[..10]);
//     println!("lm_head (前10): {:?}", &lm_head[..10]);
//     println!("block0 attn.q_proj (前10): {:?}", &q_proj[..10]);
//     // 检查不全为0
//     assert!(emb[..10].iter().any(|&x| x != 0.0), "embedding 前10项全为0");
//     assert!(
//         lm_head[..10].iter().any(|&x| x != 0.0),
//         "lm_head 前10项全为0"
//     );
//     assert!(
//         q_proj[..10].iter().any(|&x| x != 0.0),
//         "block0 attn.q_proj 前10项全为0"
//     );
// }
