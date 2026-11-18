"""Byte-level vocabulary and the program alphabet (paper Sec. 2.3, App. E)."""
from __future__ import annotations

VOCAB_SIZE = 256

BF_INSTRUCTIONS = "<>+-[].,"          # the eight Brainf*ck primitives
MACROS = "ZRLNCGHWVX"                 # ten single-byte macros (App. E, Table "macros")
END_TOKEN = "F"                       # explicit end-of-program token
PROGRAM_ALPHABET = BF_INSTRUCTIONS + MACROS + END_TOKEN   # |A| = 19
BODY_ALPHABET = BF_INSTRUCTIONS + MACROS                  # tokens allowed inside a body

START_PROGRAM = ord("S")  # prefix byte for programs (generator sequences)
START_OUTPUT = ord("O")   # prefix byte for outputs (learner sequences)
END_ID = ord(END_TOKEN)

PROGRAM_TOKEN_IDS = [ord(c) for c in PROGRAM_ALPHABET]
BODY_TOKEN_IDS = [ord(c) for c in BODY_ALPHABET]
ALPHABET_SIZE = len(PROGRAM_ALPHABET)

MACRO_EXPANSIONS = {
    "Z": "[-]",                  # clear current cell
    "R": "[->+<]",               # clear and add x into right neighbour
    "L": "[->+++<]",             # clear and add 3x into right neighbour
    "N": "[-<->]",               # clear and subtract x from left neighbour
    "C": "[->+>+<<]",            # clear and add x into the two right cells
    "G": "[>]",                  # scan right to next zero cell
    "H": "[<]",                  # scan left to next zero cell
    "W": "[[-]>+<]",             # if current != 0: increment right and clear current
    "V": "[.>]",                 # print stored string until a zero cell
    "X": "[-]++++++++++++++++",  # set cell to 16
}


def expand_macros(body: str) -> str:
    """Expand macro tokens into pure Brainf*ck."""
    return "".join(MACRO_EXPANSIONS.get(c, c) for c in body)


def body_to_str(body: bytes) -> str:
    return body.decode("latin-1")


def program_tokens(body: bytes) -> list[int]:
    """Full generator sequence for a program body: S + body + F."""
    return [START_PROGRAM, *body, END_ID]
