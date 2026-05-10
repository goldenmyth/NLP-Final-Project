def get_chat_parts(tokenizer, paragraphs, question):
    system_msg = (
        "You are a precise question-answering assistant. "
        "Answer the question using the provided context with a short phrase (1-3 words). "
        "Do not use Markdown, do not provide links, do not use full sentences. "
        "Provide only the factual answer."
    )

    prefix_text = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\nContext:"
    p_texts = [f"\n{p}" for p in paragraphs]
    suffix_text = f"\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"

    return prefix_text, p_texts, suffix_text