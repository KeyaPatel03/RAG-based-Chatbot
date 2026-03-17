from load_model import model, tokenizer 
def test_model():
    prompt = "Test: What is 3 + 5?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,    # enable sampling for more diverse outputs
        temperature=0.3
    )

    print("Output:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    test_model()
