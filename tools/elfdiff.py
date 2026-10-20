# elfdiff.py A B [N]: count the bytes where two same-size ELF files differ, per section, and show the first N spots.
import sys, subprocess, collections, re
a, b = sys.argv[1], sys.argv[2]
A, B = open(a,'rb').read(), open(b,'rb').read()
assert len(A) == len(B)
secs = []
for line in subprocess.run(['readelf','-SW',a],capture_output=True,text=True).stdout.splitlines():
    m = re.match(r'\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+[0-9a-f]+\s+([0-9a-f]+)\s+([0-9a-f]+)', line)
    if m: secs.append((m.group(1), int(m.group(2),16), int(m.group(3),16)))
def sec(o):
    for n, off, sz in secs:
        if n != 'NULL' and off <= o < off+sz and not n.startswith('.tbss') and not n.startswith('.bss'): return n
    return '?'
diff = [i for i in range(len(A)) if A[i] != B[i]]
c = collections.Counter(sec(o) for o in diff)
print(a, len(diff), 'bytes differ:', dict(c))
# show a few differing tokens in context
shown = 0; last = -100
for o in diff:
    if o - last > 16 and shown < int(sys.argv[3] if len(sys.argv) > 3 else 4):
        s = max(0, o-40); print(f'  @{o:#x} [{sec(o)}]\n    rebuilt: {A[s:o+24]!r}\n    shipped: {B[s:o+24]!r}'); shown += 1
    last = o
