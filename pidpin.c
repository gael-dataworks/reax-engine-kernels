// pidpin TARGET CMD [ARGS...]
// Runs CMD as process ID TARGET inside a fresh PID namespace (a new container): fork/reap throwaway
// children until the kernel's sequential PID counter reaches TARGET-1, then fork once more and exec CMD.
// nvcc names its temp files tmpxft_<pid hex>_..., and the linker records that name as a FILE symbol in
// .symtab, so pinning nvcc's PID is what makes the rebuilt .so byte-identical. It changes nothing else.
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: pidpin TARGET CMD [ARGS...]\n"); return 2; }
  pid_t target = (pid_t)strtol(argv[1], 0, 0);
  for (;;) {
    pid_t p = fork();
    if (p < 0) { perror("fork"); return 2; }
    if (p == 0) _exit(0);
    waitpid(p, 0, 0);
    if (p == target - 1) break;
    if (p >= target) { fprintf(stderr, "pidpin: PID counter already past %d (at %d)\n", target, p); return 2; }
  }
  pid_t c = fork();
  if (c == 0) { execvp(argv[2], argv + 2); perror("execvp"); _exit(127); }
  if (c != target) fprintf(stderr, "pidpin: WARNING child got pid %d, wanted %d\n", c, target);
  else fprintf(stderr, "pidpin: running %s as pid %d (0x%x)\n", argv[2], c, c);
  int st = 0;
  waitpid(c, &st, 0);
  return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}
