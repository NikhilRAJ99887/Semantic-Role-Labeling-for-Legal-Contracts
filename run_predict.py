from predict import analyze

examples = [
    "Meera Iyer shall deliver the user interface within three days.",
    "Delta Apps shall provide test accounts by 10 April 2027.",
    "ABC Technologies shall pay Rahul Sharma ₹50,000 upon completion of the mobile application.",
]

for text in examples:
    print("\n" + "=" * 80)
    print("INPUT:", text)
    entities, roles = analyze(text)
    print("\nWHO:", [(x["text"], x["type"]) for x in roles["WHO"]])
    print("WHAT:", [(x["text"], x["type"]) for x in roles["WHAT"]])
    print("WHEN:", [(x["text"], x["type"]) for x in roles["WHEN"]])
