// Headless Linux reference for the same pinned engine and presentation size.
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "doomgeneric.h"

static uint32_t framebuffer[800 * 600];
static volatile uint32_t checksum;
static uint64_t start;

static uint64_t milliseconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

void DG_Init(void) { start = milliseconds(); }
void DG_DrawFrame(void) {
    for (unsigned y = 0; y < DOOMGENERIC_RESY; ++y) {
        memcpy(framebuffer + (y + 100) * 800 + 80,
               DG_ScreenBuffer + y * DOOMGENERIC_RESX, DOOMGENERIC_RESX * 4);
    }
    checksum ^= framebuffer[40000 + DG_GetTicksMs() % 300000];
}
void DG_SleepMs(uint32_t milliseconds) {
    struct timespec delay = {milliseconds / 1000, (milliseconds % 1000) * 1000000};
    nanosleep(&delay, NULL);
}
uint32_t DG_GetTicksMs(void) { return (uint32_t)(milliseconds() - start); }
int DG_GetKey(int *pressed, unsigned char *key) { (void)pressed; (void)key; return 0; }
void DG_SetWindowTitle(const char *title) { (void)title; }
void doom_engine_report(const char *message) { printf("LINUX_DOOM_ENGINE_REPORT %s\n", message); fflush(stdout); }
int main(int argc, char **argv) {
    doomgeneric_Create(argc, argv);
    for (;;) doomgeneric_Tick();
}
