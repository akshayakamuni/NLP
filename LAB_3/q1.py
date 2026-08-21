def dfa(word):
    state = "q0"

    for ch in word:

        if state == "q0":
            if 'a' <= ch <= 'z':
                state = "q1"
            else:
                state = "q_dead"

        elif state == "q1":
            if 'a' <= ch <= 'z':
                state = "q1"
            else:
                state = "q_dead"

        elif state == "q_dead":
            state = "q_dead"

    # q1 is the only accepting state
    if state == "q1":
        return "Accepted"
    else:
        return "Not Accepted"


# Test
words = [
    "cat",
    "dog",
    "a",
    "zebra",
    "dog1",
    "1dog",
    "DogHouse",
    "Dog_house",
    " cats"
]

for word in words:
    print(word, "->", dfa(word))
