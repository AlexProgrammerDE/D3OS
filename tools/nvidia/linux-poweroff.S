/* Linux reference guest PID 1: power off without returning to pageable shell code. */
.section .text
.global _start
.type _start, @function
_start:
    mov $169, %eax                  /* __NR_reboot */
    mov $0xfee1dead, %edi           /* LINUX_REBOOT_MAGIC1 */
    mov $0x28121969, %esi           /* LINUX_REBOOT_MAGIC2 */
    mov $0x4321fedc, %edx           /* LINUX_REBOOT_CMD_POWER_OFF */
    xor %r10d, %r10d
    syscall
    /* A successful poweroff does not return. Report a rejected syscall. */
    mov $60, %eax                   /* __NR_exit */
    mov $1, %edi
    syscall
.size _start, .-_start
.section .note.GNU-stack,"",@progbits
