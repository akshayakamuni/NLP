from automathon import DFA
from graphviz import Digraph
import os


# ============================================================
# PART 1
# DFA FOR VALID SIMPLIFIED ENGLISH WORDS
# ============================================================

def create_dfa():

    # --------------------------------------------------------
    # STATES
    # --------------------------------------------------------

    Q = {
        "q0",
        "q1",
        "q_dead"
    }

    # --------------------------------------------------------
    # INPUT ALPHABET
    # Only lowercase English letters
    # --------------------------------------------------------

    sigma = set("abcdefghijklmnopqrstuvwxyz")

    # --------------------------------------------------------
    # TRANSITION FUNCTION
    # --------------------------------------------------------

    delta = {
        "q0": {},
        "q1": {},
        "q_dead": {}
    }

    # q0:
    # First character must be lowercase

    for ch in sigma:
        delta["q0"][ch] = "q1"

    # q1:
    # Remaining characters must be lowercase

    for ch in sigma:
        delta["q1"][ch] = "q1"

    # q_dead:
    # Once invalid, remain invalid

    for ch in sigma:
        delta["q_dead"][ch] = "q_dead"

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    initial_state = "q0"

    # --------------------------------------------------------
    # FINAL / ACCEPTING STATE
    # --------------------------------------------------------

    F = {"q1"}

    # --------------------------------------------------------
    # CREATE DFA
    # --------------------------------------------------------

    dfa = DFA(
        Q,
        sigma,
        delta,
        initial_state,
        F
    )

    return dfa


# ============================================================
# DFA VALIDATION FUNCTION
# ============================================================

def is_valid_word(word, dfa):

    word = word.lower()

    # Empty string is invalid
    if len(word) == 0:
        return False

    # First character must be lowercase a-z
    if not ("a" <= word[0] <= "z"):
        return False

    # Every character must be lowercase a-z
    if not all("a" <= ch <= "z" for ch in word):
        return False

    # DFA validation
    return dfa.accept(word)


# ============================================================
# DFA TESTING
# ============================================================

def test_dfa(dfa):

    print("\n")
    print("=" * 60)
    print("PART 1: DFA TESTING")
    print("=" * 60)

    test_words = [
        "cat",
        "dog",
        "a",
        "zebra",
        "hello",

        "dog1",
        "1dog",
        "DogHouse",
        "Dog_house",
        " cats",
        "cat!",
        "hello world",
        "",
        "742",
        "75%",
        "80%",
        "80's",
        "9/32",
        "a/3"
    ]

    for word in test_words:

        accepted = is_valid_word(word, dfa)

        if accepted:
            print(f"{word!r:20} -> Accepted")
        else:
            print(f"{word!r:20} -> Not Accepted")


# ============================================================
# DFA VISUALIZATION
# ============================================================

def visualize_dfa(dfa):

    print("\nCreating DFA visualization...")

    os.makedirs("output", exist_ok=True)

    dfa.view(
        "output/DFA_English_Words"
    )

    print("DFA visualization created.")
    print("Check the output folder.")


# ============================================================
# PART 2
# FST / MORPHOLOGICAL ANALYZER
# ============================================================


# ============================================================
# LOAD BROWN NOUNS
# ============================================================

def load_nouns(filename):

    nouns = set()

    try:

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                word = line.strip().lower()

                if not word:
                    continue

                # ------------------------------------------------
                # IMPORTANT:
                # Only keep valid simplified English words.
                #
                # This prevents entries such as:
                # 742
                # 75%
                # 9/32
                # a/3
                # 80's
                # ------------------------------------------------

                if all("a" <= ch <= "z" for ch in word):

                    nouns.add(word)

    except FileNotFoundError:

        print("\nERROR:")
        print("brown_nouns.txt was not found.")

        return set()

    return nouns


# ============================================================
# DETERMINE MORPHOLOGICAL CLASS
# ============================================================

def get_plural_class(word):

    # --------------------------------------------------------
    # Y REPLACEMENT
    #
    # try -> tries
    # cry -> cries
    # --------------------------------------------------------

    if word.endswith("y"):
        return "Y"

    # --------------------------------------------------------
    # E INSERTION
    #
    # fox -> foxes
    # box -> boxes
    # buzz -> buzzes
    # watch -> watches
    # dish -> dishes
    # class -> classes
    # --------------------------------------------------------

    if (
        word.endswith("s")
        or word.endswith("z")
        or word.endswith("x")
        or word.endswith("ch")
        or word.endswith("sh")
    ):
        return "ES"

    # --------------------------------------------------------
    # NORMAL S ADDITION
    #
    # cat -> cats
    # bag -> bags
    # dog -> dogs
    # accomplishment -> accomplishments
    # --------------------------------------------------------

    return "S"


# ============================================================
# FST ANALYZER
# ============================================================

def analyze_word(word, nouns):

    word = word.lower()

    # ========================================================
    # CASE 0: VALID WORD CHECK
    # ========================================================

    # Empty word
    if len(word) == 0:
        return "Invalid Word"

    # First character must be lowercase letter
    if not ("a" <= word[0] <= "z"):
        return "Invalid Word"

    # Every character must be lowercase letter
    if not all("a" <= ch <= "z" for ch in word):
        return "Invalid Word"

    # ========================================================
    # CASE 1: Y REPLACEMENT PLURAL
    #
    # tries -> try+N+PL
    # cries -> cry+N+PL
    #
    # This MUST be checked before singular membership.
    # ========================================================

    if word.endswith("ies"):

        root = word[:-3] + "y"

        if root in nouns:

            if get_plural_class(root) == "Y":

                return root + "+N+PL"

    # ========================================================
    # CASE 2: E INSERTION PLURAL
    #
    # foxes -> fox+N+PL
    # watches -> watch+N+PL
    # dishes -> dish+N+PL
    # classes -> class+N+PL
    #
    # This MUST be checked before singular membership.
    # ========================================================

    if word.endswith("es"):

        root = word[:-2]

        if root in nouns:

            if get_plural_class(root) == "ES":

                return root + "+N+PL"

    # ========================================================
    # CASE 3: NORMAL S ADDITION PLURAL
    #
    # bags -> bag+N+PL
    # cats -> cat+N+PL
    # dogs -> dog+N+PL
    # accomplishments -> accomplishment+N+PL
    #
    # This MUST be checked before singular membership.
    # ========================================================

    if word.endswith("s"):

        root = word[:-1]

        if root in nouns:

            if get_plural_class(root) == "S":

                return root + "+N+PL"

    # ========================================================
    # CASE 4: SINGULAR
    #
    # Only reached if the word was NOT recognized as a
    # valid plural.
    # ========================================================

    if word in nouns:

        return word + "+N+SG"

    # ========================================================
    # CASE 5: INVALID
    # ========================================================

    return "Invalid Word"


# ============================================================
# TEST FST
# ============================================================

def test_fst(nouns):

    print("\n")
    print("=" * 60)
    print("PART 2: FST TESTING")
    print("=" * 60)

    test_words = [

        # ----------------------------------------------------
        # Singular
        # ----------------------------------------------------

        "fox",
        "watch",
        "try",
        "bag",
        "cat",
        "dog",
        "accomplishment",

        # ----------------------------------------------------
        # Correct plurals
        # ----------------------------------------------------

        "foxes",
        "watches",
        "tries",
        "bags",
        "cats",
        "dogs",
        "accomplishments",

        # ----------------------------------------------------
        # Incorrect plurals
        # ----------------------------------------------------

        "foxs",
        "watchs",
        "trys",

        # ----------------------------------------------------
        # Invalid inputs
        # ----------------------------------------------------

        "742",
        "75%",
        "80%",
        "80's",
        "9/32",
        "a/3",
        "Dog",
        "hello!",
        "hello123"
    ]

    for word in test_words:

        result = analyze_word(
            word,
            nouns
        )

        print(
            f"{word:20} -> {result}"
        )


# ============================================================
# PROCESS ALL BROWN NOUNS
# ============================================================

def process_brown_corpus(nouns):

    print("\n")
    print("=" * 60)
    print("PROCESSING BROWN CORPUS")
    print("=" * 60)

    os.makedirs("output", exist_ok=True)

    output_file = (
        "output/brown_fst_results.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        for noun in sorted(nouns):

            result = analyze_word(
                noun,
                nouns
            )

            file.write(
                f"{noun} = {result}\n"
            )

    print(
        "Number of nouns:",
        len(nouns)
    )

    print(
        "Results saved to:",
        output_file
    )


# ============================================================
# FST VISUALIZATION
# ============================================================

def visualize_fst():

    print("\nCreating FST visualization...")

    os.makedirs("output", exist_ok=True)

    dot = Digraph(
        "Morphological_FST",
        format="png"
    )

    # Left to right
    dot.attr(
        rankdir="LR"
    )

    # ========================================================
    # STATES
    # ========================================================

    dot.node(
        "q0",
        "q0\nStart",
        shape="circle"
    )

    dot.node(
        "qN",
        "qN\nNormal",
        shape="circle"
    )

    dot.node(
        "qES",
        "qES\nE-Insertion",
        shape="circle"
    )

    dot.node(
        "qY",
        "qY\nY-Replacement",
        shape="circle"
    )

    dot.node(
        "qF",
        "qF\nFinal",
        shape="doublecircle"
    )

    dot.node(
        "qD",
        "qD\nInvalid",
        shape="circle"
    )

    # ========================================================
    # START
    # ========================================================

    dot.edge(
        "q0",
        "qN",
        label="letter : same"
    )

    # ========================================================
    # NORMAL CHARACTERS
    # ========================================================

    dot.edge(
        "qN",
        "qN",
        label="normal letter : same"
    )

    # ========================================================
    # E-INSERTION CLASS
    # ========================================================

    dot.edge(
        "qN",
        "qES",
        label="s/x/z : same"
    )

    dot.edge(
        "qN",
        "qES",
        label="ch/sh : same"
    )

    # ========================================================
    # Y-REPLACEMENT CLASS
    # ========================================================

    dot.edge(
        "qN",
        "qY",
        label="y : y"
    )

    # ========================================================
    # SINGULAR
    # ========================================================

    dot.edge(
        "qN",
        "qF",
        label="END : +N+SG"
    )

    # ========================================================
    # NORMAL PLURAL
    # ========================================================

    dot.edge(
        "qN",
        "qF",
        label="s : +N+PL"
    )

    # ========================================================
    # E-INSERTION PLURAL
    # ========================================================

    dot.edge(
        "qES",
        "qF",
        label="es : +N+PL"
    )

    # ========================================================
    # Y-REPLACEMENT PLURAL
    # ========================================================

    dot.edge(
        "qY",
        "qF",
        label="ies : +N+PL"
    )

    # ========================================================
    # INVALID FORMS
    # ========================================================

    dot.edge(
        "qES",
        "qD",
        label="s : Invalid"
    )

    dot.edge(
        "qY",
        "qD",
        label="s : Invalid"
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_path = dot.render(
        "output/FST_Morphological",
        cleanup=True
    )

    print(
        "FST visualization created:"
    )

    print(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # PART 1: DFA
    # ========================================================

    print("\n")
    print("#" * 60)
    print("PART 1: DFA")
    print("#" * 60)

    dfa = create_dfa()

    print(
        "\nDFA valid:",
        dfa.is_valid()
    )

    test_dfa(dfa)

    visualize_dfa(dfa)

    # ========================================================
    # PART 2: FST
    # ========================================================

    print("\n")
    print("#" * 60)
    print("PART 2: FST")
    print("#" * 60)

    nouns = load_nouns(
        "brown_nouns.txt"
    )

    if len(nouns) == 0:

        print(
            "\nNo nouns were loaded."
        )

        print(
            "Make sure brown_nouns.txt is "
            "in the same folder as this Python file."
        )

        return

    print(
        "\nLoaded",
        len(nouns),
        "valid Brown corpus nouns."
    )

    # ========================================================
    # TEST EXAMPLES
    # ========================================================

    test_fst(nouns)

    # ========================================================
    # PROCESS COMPLETE CORPUS
    # ========================================================

    process_brown_corpus(nouns)

    # ========================================================
    # VISUALIZE FST
    # ========================================================

    visualize_fst()

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n")
    print("=" * 60)
    print("ALL TASKS COMPLETED")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()