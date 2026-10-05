/* edge_inference.c
 * Pure C inference for the trained PPO lander policy - no Python, no
 * numpy, no dynamic allocation, no external dependencies beyond libm.
 * This is what would actually run on a CubeSat/SmallSat OBC or any
 * other ARM/edge processor once cross-compiled for that target.
 *
 * Usage: ./edge_infer x y vx vy fuel
 *   Prints the 4 action probabilities and the chosen (argmax) action.
 *
 * HONEST SCOPE: compiled and run natively (x86) here to verify numeric
 * correctness against the Python/numpy policy bit-for-bit (within
 * float32 precision) - this is the proof-of-correctness step. ARM
 * cross-compilation (e.g. via arm-none-eabi-gcc) and int8 quantization
 * are the next steps, not done in this file.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "model_weights.h"

static void forward(const float state_raw[STATE_DIM], float probs_out[ACTION_DIM], int *argmax_out) {
    float state[STATE_DIM];
    for (int i = 0; i < STATE_DIM; i++) {
        state[i] = state_raw[i] / STATE_SCALE[i];
    }

    float h1[HIDDEN_DIM];
    for (int j = 0; j < HIDDEN_DIM; j++) {
        float z = B1[j];
        for (int i = 0; i < STATE_DIM; i++) {
            z += state[i] * W1[i * HIDDEN_DIM + j];
        }
        h1[j] = z > 0.0f ? z : 0.0f;  /* ReLU */
    }

    float logits[ACTION_DIM];
    float max_logit = -1e30f;
    for (int k = 0; k < ACTION_DIM; k++) {
        float z = B2[k];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            z += h1[j] * W2[j * ACTION_DIM + k];
        }
        logits[k] = z;
        if (z > max_logit) max_logit = z;
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < ACTION_DIM; k++) {
        probs_out[k] = expf(logits[k] - max_logit);
        sum_exp += probs_out[k];
    }
    int argmax = 0;
    float best = -1.0f;
    for (int k = 0; k < ACTION_DIM; k++) {
        probs_out[k] /= sum_exp;
        if (probs_out[k] > best) { best = probs_out[k]; argmax = k; }
    }
    *argmax_out = argmax;
}

int main(int argc, char **argv) {
    if (argc != STATE_DIM + 1) {
        fprintf(stderr, "Usage: %s x y vx vy fuel\n", argv[0]);
        return 1;
    }
    float state[STATE_DIM];
    for (int i = 0; i < STATE_DIM; i++) {
        state[i] = (float)atof(argv[i + 1]);
    }

    float probs[ACTION_DIM];
    int action;
    forward(state, probs, &action);

    printf("probs:");
    for (int k = 0; k < ACTION_DIM; k++) printf(" %.8f", probs[k]);
    printf("\naction: %d\n", action);
    return 0;
}
