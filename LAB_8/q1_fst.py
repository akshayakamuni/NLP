
def pluralize(noun):
    special_endings = ("s", "x", "z", "ch", "sh")

    if noun.lower().endswith(special_endings):
        return noun + "es"

    return noun + "s"


def analyze(noun):
    return {
        "singular": noun + "+N+SG",
        "plural": pluralize(noun) + "+N+PL"
    }


nouns = ["cat", "book", "bus", "box", "buzz", "watch", "dish"]

for noun in nouns:
    result = analyze(noun)
    print("Noun:", noun)
    print("Singular:", result["singular"])
    print("Plural:", result["plural"])
    print("-" * 30)
