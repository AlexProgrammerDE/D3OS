// Preserve engine error handling while making timedemo results available to
// serial automation. The pinned engine reports a completed timedemo via I_Error.
#include <stdarg.h>
#include <stdio.h>
#include "../doomgeneric/doomgeneric/z_zone.h"

extern void doom_engine_report(const char *message);
extern void __real_I_Error(char *format, ...);

// Use the pinned engine's enum rather than duplicating its purge tag in Rust.
void doom_purge_texture_cache(void) {
    Z_FreeTags(PU_CACHE, PU_CACHE);
}

void __wrap_I_Error(char *format, ...) {
    char message[512];
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);
    doom_engine_report(message);
    __real_I_Error("%s", message);
}
