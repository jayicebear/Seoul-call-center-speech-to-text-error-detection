def correct_sentence(asr_sentence):
    input_text = f"{asr_sentence} →"
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_length=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(correct_sentence("오늘 날씨는 좋습니디"))
