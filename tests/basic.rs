use qwen3_infer::tokenizer::tokenizer::QwenTokenizer;

#[test]
fn test_tokenizer() {
    let tokenizer =
        QwenTokenizer::from_vocab_and_merges("Qwen3-0.6B/vocab.json", "Qwen3-0.6B/merges.txt")
            .unwrap();

    let tokens = tokenizer.encode("i am a student").unwrap();
    println!("{:?}", tokens);
    let text = tokenizer.decode(&tokens).unwrap();
    assert_eq!(text, "iam ast udent");
}
