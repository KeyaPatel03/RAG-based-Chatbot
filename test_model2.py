from load_model import model, tokenizer

def ask(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,     # enable sampling for more diverse outputs
        temperature=0.3,         # lower = more deterministic reasoning 
        top_p=0.9
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":

    reasoning_questions = [
        "A farmer has 17 goats. 8 die. How many are left? Explain your reasoning.",
        "If you move 3 apples from a basket of 10 into a second basket with 4, which basket has more apples now?",
        "Why does the Earth experience seasons? Explain step by step.",
        "A train leaves station A at 3 PM going 60 km/h. Another train leaves station B at 4 PM going 80 km/h toward A. Explain how to calculate when they meet.",
        "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Explain your reasoning.",
        "You have two ropes that each burn in 60 minutes but not at a constant rate. How do you measure exactly 45 minutes?"
    ]

    for q in reasoning_questions:
        print("\n----------------------")
        print("QUESTION:", q)
        print("ANSWER:")
        print(ask(q))
