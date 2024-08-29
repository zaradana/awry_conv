def tokenize_function(examples, tokenizer):
    
    if not isinstance(examples['text_a'], (str, list, tuple)) or not isinstance(examples['text_b'], (str, list, tuple)):
        print(f"text_a: {examples['text_a']}")
        print(f"text_b: {examples['text_b']}")
        raise ValueError("Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.")
    
    return tokenizer(examples['text_a'], examples['text_b'], padding='max_length', truncation=True)