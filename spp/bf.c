/* Brainf*ck-like universal machine (paper App. E).
 *
 *  - circular tape of `tape_size` byte cells, arithmetic mod 256
 *  - unmatched brackets are no-ops
 *  - ',' reads an i.i.d. uniform random byte from the input tape (PRNG)
 *  - execution stops at the step budget, the end of the program, or the T-th
 *    emitted byte; the output is zero-padded to T bytes
 *  - ten single-byte macros are expanded to pure Brainf*ck before execution
 *  - reports the maximum dynamic loop depth (used as a MAP-Elites descriptor)
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static const char *macro_expansion(unsigned char c) {
    switch (c) {
        case 'Z': return "[-]";
        case 'R': return "[->+<]";
        case 'L': return "[->+++<]";
        case 'N': return "[-<->]";
        case 'C': return "[->+>+<<]";
        case 'G': return "[>]";
        case 'H': return "[<]";
        case 'W': return "[[-]>+<]";
        case 'V': return "[.>]";
        case 'X': return "[-]++++++++++++++++";
        default:  return NULL;
    }
}

/* Number of bytes bf_expand will write for this program (exact; the longest macro, X, is 19 bytes). */
int bf_expanded_len(const unsigned char *prog, int plen) {
    int n = 0;
    for (int i = 0; i < plen; i++) {
        unsigned char c = prog[i];
        if (c == 'F') break;
        const char *e = macro_expansion(c);
        n += e ? (int)strlen(e) : 1;
    }
    return n;
}

/* Expand macros; stops at 'F'. Returns expanded length. `out` must hold bf_expanded_len() bytes. */
int bf_expand(const unsigned char *prog, int plen, unsigned char *out) {
    int n = 0;
    for (int i = 0; i < plen; i++) {
        unsigned char c = prog[i];
        if (c == 'F') break;
        const char *e = macro_expansion(c);
        if (e) { int l = (int)strlen(e); memcpy(out + n, e, l); n += l; }
        else { out[n++] = c; }
    }
    return n;
}

/* Run one program. Returns the number of executed instructions. */
int64_t bf_run(const unsigned char *prog, int plen, int tape_size, int64_t step_budget,
               int T, uint64_t seed, unsigned char *out, int *n_emitted, int *max_depth) {
    int cap = bf_expanded_len(prog, plen) + 1;
    unsigned char *code = (unsigned char *)malloc(cap);
    int n = bf_expand(prog, plen, code);
    int *jump = (int *)malloc(sizeof(int) * (n + 1));
    int *stack = (int *)malloc(sizeof(int) * (n + 1));
    int sp = 0;
    for (int i = 0; i < n; i++) jump[i] = -1;
    for (int i = 0; i < n; i++) {
        if (code[i] == '[') stack[sp++] = i;
        else if (code[i] == ']' && sp > 0) { int j = stack[--sp]; jump[i] = j; jump[j] = i; }
    }
    unsigned char *tape = (unsigned char *)calloc((size_t)tape_size, 1);
    memset(out, 0, (size_t)T);
    uint64_t rng = seed;
    int head = 0, pc = 0, emitted = 0, depth = 0, maxd = 0;
    int64_t steps = 0;
    while (pc < n && steps < step_budget && emitted < T) {
        unsigned char c = code[pc];
        steps++;
        switch (c) {
            case '>': head = (head + 1 == tape_size) ? 0 : head + 1; break;
            case '<': head = (head == 0) ? tape_size - 1 : head - 1; break;
            case '+': tape[head]++; break;
            case '-': tape[head]--; break;
            case '.': out[emitted++] = tape[head]; break;
            case ',': tape[head] = (unsigned char)(splitmix64(&rng) >> 56); break;
            case '[':
                if (jump[pc] >= 0) {
                    if (tape[head] == 0) pc = jump[pc];
                    else { depth++; if (depth > maxd) maxd = depth; }
                }
                break;
            case ']':
                if (jump[pc] >= 0) {
                    if (tape[head] != 0) pc = jump[pc];
                    else depth--;
                }
                break;
            default: break;
        }
        pc++;
    }
    *n_emitted = emitted;
    *max_depth = maxd;
    free(tape); free(stack); free(jump); free(code);
    return steps;
}

/* Run a batch of programs stored contiguously: program i is progs[offsets[i]:offsets[i+1]]. */
void bf_run_batch(const unsigned char *progs, const int64_t *offsets, int n,
                  int tape_size, int64_t step_budget, int T, uint64_t seed,
                  unsigned char *out, int *n_emitted, int *max_depth, int64_t *steps) {
    #pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < n; i++) {
        uint64_t s = seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1));
        steps[i] = bf_run(progs + offsets[i], (int)(offsets[i + 1] - offsets[i]), tape_size,
                          step_budget, T, s, out + (int64_t)i * T, n_emitted + i, max_depth + i);
    }
}
