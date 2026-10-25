/* ondrik: the program of the read-the-manual tasks (docs/training.md,
 * "The agent image"). The image compiles it in a build stage and installs
 * only the binary, on the path, executable but not readable by the
 * model's user, with its manual, /usr/share/doc/ondrik/manual.txt, which a
 * task may remove (`absent`): without the manual, nothing in the image
 * documents it.
 *
 * It prints the seal of a text and the knot of two numbers, by rules of
 * its own, so only the program gives them. It prints no help of its own:
 * a call it cannot read (an unknown command or option, --help included, a
 * missing or extra argument, a number that is not a whole number) names
 * the manual on standard error and exits with code 2.
 *
 *   seal: sha256("ondrik press\0" TEXT), then turns - 1 times
 *         sha256("ondrik turn\0" digest); its first 50 bits as 10 signs of
 *         ALPHABET, 5 bits each, the first bits first
 *   knot: sha256("ondrik tie\0" A "\0" B), or "ondrik give" for the slack
 *         knot, A and B in decimal without leading zeros; the digest, a
 *         big-endian number, modulo 10^6, written with 6 digits
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MANUAL "/usr/share/doc/ondrik/manual.txt"
static const char ALPHABET[] = "0123456789abcdefghjkmnpqrstvwxyz"; /* the digits and the letters but i, l, o, u */
enum { SEAL = 10, KNOT = 1000000, MOST_TURNS = 1000000 };

/* -- SHA-256 (FIPS 180-4) ---------------------------------------------------- */

typedef struct { uint32_t state[8]; uint64_t length; unsigned char block[64]; size_t used; } Sha256;

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

static uint32_t rotate(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void compress(Sha256 *h, const unsigned char *block) {
    uint32_t w[64], s[8];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)block[4 * i] << 24 | (uint32_t)block[4 * i + 1] << 16 | (uint32_t)block[4 * i + 2] << 8
               | block[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotate(w[i - 15], 7) ^ rotate(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotate(w[i - 2], 17) ^ rotate(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    memcpy(s, h->state, sizeof s);
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = s[7] + (rotate(s[4], 6) ^ rotate(s[4], 11) ^ rotate(s[4], 25)) + ((s[4] & s[5]) ^ (~s[4] & s[6]))
                      + K[i] + w[i];
        uint32_t t2 = (rotate(s[0], 2) ^ rotate(s[0], 13) ^ rotate(s[0], 22)) + ((s[0] & s[1]) ^ (s[0] & s[2])
                                                                                ^ (s[1] & s[2]));
        memmove(s + 1, s, 7 * sizeof s[0]);
        s[4] += t1;
        s[0] = t1 + t2;
    }
    for (int i = 0; i < 8; i++)
        h->state[i] += s[i];
}

static void start(Sha256 *h) {
    static const uint32_t first[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(h->state, first, sizeof first);
    h->length = 0;
    h->used = 0;
}

static void add(Sha256 *h, const void *data, size_t size) {
    const unsigned char *bytes = data;
    h->length += size;
    while (size) {
        size_t room = 64 - h->used, taken = size < room ? size : room;
        memcpy(h->block + h->used, bytes, taken);
        h->used += taken, bytes += taken, size -= taken;
        if (h->used == 64) {
            compress(h, h->block);
            h->used = 0;
        }
    }
}

static void finish(Sha256 *h, unsigned char digest[32]) {
    uint64_t bits = h->length * 8;
    unsigned char end[8];
    add(h, "\x80", 1);
    while (h->used != 56)
        add(h, "", 1);
    for (int i = 0; i < 8; i++)
        end[i] = (unsigned char)(bits >> (56 - 8 * i));
    add(h, end, 8);
    for (int i = 0; i < 32; i++)
        digest[i] = (unsigned char)(h->state[i / 4] >> (24 - 8 * (i % 4)));
}

/* -- ondrik ----------------------------------------------------------------- */

static void unread(const char *why, const char *word) {
    fprintf(stderr, "ondrik: %s", why);
    if (word)
        fprintf(stderr, " '%s'", word);
    fprintf(stderr, "; see %s\n", MANUAL);
    exit(2);
}

/* The whole number word holds, in decimal without leading zeros. */
static const char *number(const char *word, const char *what) {
    if (!*word)
        unread(what, word);
    for (const char *c = word; *c; c++)
        if (*c < '0' || *c > '9')
            unread(what, word);
    while (word[0] == '0' && word[1])
        word++;
    return word;
}

static void seal(const char *text, long turns) {
    Sha256 h;
    unsigned char digest[32];
    start(&h);
    add(&h, "ondrik press", 13); /* with its NUL */
    add(&h, text, strlen(text));
    finish(&h, digest);
    for (long i = 1; i < turns; i++) {
        start(&h);
        add(&h, "ondrik turn", 12);
        add(&h, digest, 32);
        finish(&h, digest);
    }
    uint64_t top = 0; /* the first 64 bits */
    for (int i = 0; i < 8; i++)
        top = top << 8 | digest[i];
    char signs[SEAL + 1];
    for (int place = 0; place < SEAL; place++)
        signs[place] = ALPHABET[top >> (64 - 5 * (place + 1)) & 31];
    signs[SEAL] = 0;
    puts(signs);
}

static void knot(const char *first, const char *second, int give) {
    Sha256 h;
    unsigned char digest[32];
    start(&h);
    add(&h, give ? "ondrik give" : "ondrik tie", give ? 12 : 11); /* with its NUL */
    add(&h, first, strlen(first) + 1);
    add(&h, second, strlen(second));
    finish(&h, digest);
    unsigned long rest = 0;
    for (int i = 0; i < 32; i++)
        rest = (rest * 256 + digest[i]) % KNOT;
    printf("%06lu\n", rest);
}

int main(int argc, char **argv) {
    if (argc < 2)
        unread("no command", NULL);
    const char *command = argv[1];
    char **rest = argv + 2;
    int left = argc - 2;
    if (!strcmp(command, "press")) {
        long turns = 1;
        if (left && !strcmp(rest[0], "--turns")) {
            if (left < 2)
                unread("--turns needs a number", NULL);
            const char *count = number(rest[1], "--turns must be a whole number, not");
            if (!strcmp(count, "0"))
                unread("--turns must be 1 or more", NULL);
            if (strlen(count) > 7 || (turns = atol(count)) > MOST_TURNS)
                unread("--turns is more than 1000000:", count);
            rest += 2, left -= 2;
        }
        if (left && rest[0][0] == '-')
            unread("no option", rest[0]);
        if (left != 1)
            unread("press takes one text", NULL);
        seal(rest[0], turns);
        return 0;
    }
    if (!strcmp(command, "tie")) {
        int give = left && !strcmp(rest[0], "--give");
        rest += give, left -= give;
        if (left != 2)
            unread("tie takes two numbers", NULL);
        const char *first = number(rest[0], "a number must be a whole number, not");
        const char *second = number(rest[1], "a number must be a whole number, not");
        knot(first, second, give);
        return 0;
    }
    unread("no command", command);
}
