// Preserve engine error handling while making timedemo results available to
// serial automation. The pinned engine reports a completed timedemo via I_Error.
#include <stdarg.h>
#include <stdio.h>

extern void doom_engine_report(const char *message);
extern void __real_I_Error(char *format, ...);

void __wrap_I_Error(char *format, ...) {
    char message[512];
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);
    doom_engine_report(message);
    __real_I_Error("%s", message);
}
