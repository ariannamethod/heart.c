/* heart.c — core/selector.c
 *
 * Field-driven next-speaker selector. argmax cos(voice_state,
 * field_state) over {VOICE_YENT, VOICE_ARIANNA, VOICE_LEO,
 * VOICE_DOE} (ARCHITECTURE.md §1).
 *
 * STATUS: skeleton. Used by runtime/heart_main.c at every turn
 * boundary to pick the next active organism.
 */
#include "field_clock.h"

int heart_selector_pick_next(const HeartFieldClock *fc)
{
    (void)fc;
    /* TODO Phase 7: for each peer voice, compute resonance against
     *               field's primary_phase[v] + voice_coupling row,
     *               return argmax. */
    return VOICE_YENT;
}
