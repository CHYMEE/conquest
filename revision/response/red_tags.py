"""Which reviewer comments each red passage of revision.tex answers.

revision.tex no longer carries the \\rev{Rx.y} tags; revision_tagged.tex is the last tagged version
(git show 1f4398a~1:paper/revision.tex). Red passages occur in the same order in both files, so
the k-th red passage of revision.tex is the k-th red passage of the tagged file.

span_tags(text)  -> list of (position in text of '\\textcolor{red}{', set of tags), one per red passage
float_tags()     -> {float label: set of tags} from the tags inside each float (usually its caption)
A red passage without its own tags (mostly an equation) takes the tags of the red sentence that
introduces it when only equation markup separates the two, and otherwise the tags of the next
tagged passage. OVERRIDE fixes the one passage that neither rule assigns correctly.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BS = chr(92)
RED = re.compile(re.escape(BS + 'textcolor{') + r'(?:red|r)\}\{')
TAG = re.compile(re.escape(BS + 'rev{') + r'(R\d\.\d+(?:,\s*R\d\.\d+)*)\}')
FLOAT = re.compile(r'\\begin\{((?:table|figure|algorithm)\*?)\}.*?\\end\{\1\}', re.S)
# the latency and memory entries of the evaluation-metric set answer the cost comment
OVERRIDE = {BS + 'mathcal{L}_{' + BS + 'rm infer},' + BS + ';' + BS + 'mathcal{M}}': {'R1.10'}}
# sentences inside a red passage that also answer another review: (review, first words, last words)
EXTRA = [
    ('R2.3', 'CONQUEST is quantum-' + BS + 'emph{inspired}', 'not a measure of model\ncapacity.'),
    ('R2.3', 'A\nclassical binary PSO selector provides', 'same budget of 620 fitness evaluations.'),
    ('R1.11', 'Training runs\nfor at most 35 epochs.', 'down to $10^{-6}$.'),
    ('R1.4', 'The fused representation therefore has', '(69 static and 17 context).'),
    # the cost categories of Section III-D also define the units asked for in Review 2.4
    ('R2.4', BS + 'textit{Offline training cost} is incurred', 'both measured as wall-clock time.'),
    ('R2.4', BS + 'textit{Online deployment cost} is incurred', 'The per-window latency is'),
    ('R2.4', BS + 'mathcal{L}_{' + BS + 'rm infer} =\n' + BS + 'mathcal{L}_{' + BS + 'rm ctx}',
     BS + 'mathcal{L}_{' + BS + 'rm clf},'),
    ('R2.4', 'where $' + BS + 'mathcal{L}_{' + BS + 'rm ctx}$ is the', 'one forward pass of the classifier.'),
]


def extra_ranges(text, review):
    """(start, end) in text of the EXTRA sentences that answer this review."""
    out = []
    for r, a, b in EXTRA:
        if r != review:
            continue
        i = text.find(a)
        j = text.find(b, i)
        if i < 0 or j < 0:
            raise SystemExit('EXTRA sentence not found: %r ... %r' % (a, b))
        out.append((i, j + len(b)))
    return out


def blank_comments(t):
    """Comment lines replaced by blank lines, as build_response does."""
    return '\n'.join('' if l.lstrip().startswith('%') else l for l in t.split('\n'))


def _tagged():
    t = blank_comments(open(os.path.join(HERE, 'revision_tagged.tex'), encoding='utf-8').read())
    return t[t.find(BS + 'begin{document}'):]


def _tags(s):
    return {x.strip() for m in TAG.finditer(s) for x in m.group(1).split(',')}


def group_end(t, i):
    depth, k = 0, i
    while k < len(t):
        c = t[k]
        if c == '\\':
            k += 2; continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return k + 1
        k += 1
    return len(t)


EQ_GLUE = re.compile(r'(\\begin\{(equation|align)\}|\\end\{(equation|align)\}|\\label\{[^}]*\}|\\nonumber|\\\\|&|[\s{}])*')


def _spans_with_tags(t):
    spans = []
    for m in RED.finditer(t):
        k = m.start()
        lead = re.search(r'((?:\\rev\{[^}]*\})+)\s*$', t[max(0, k - 60):k])
        spans.append([m.start(), group_end(t, m.end() - 1), _tags(lead.group(1)) if lead else set()])
    for n, sp in enumerate(spans):
        if sp[2]:
            continue
        body = t[sp[0] + len('\\textcolor{red}{'):sp[0] + 80]
        for key, tags in OVERRIDE.items():
            if body.startswith(key):
                sp[2] = set(tags)
        if sp[2]:
            continue
        prev = next((spans[j] for j in range(n - 1, -1, -1) if spans[j][2]), None)
        if prev and EQ_GLUE.fullmatch(t[prev[1]:sp[0]]):
            sp[2] = set(prev[2])
        else:
            nxt = next((s for s in spans[n + 1:] if s[2] and _tags(t[max(0, s[0] - 60):s[0]])), None)
            sp[2] = set(nxt[2]) if nxt else set()
    return [(s[0], s[2]) for s in spans]


def _snippet(t, k):
    return re.sub(r'\s+', ' ', t[k:k + 60]).replace(BS + 'textcolor{r}{', BS + 'textcolor{red}{')


def span_tags(text):
    tagged = _tagged()
    old = _spans_with_tags(tagged)
    cur = [m.start() for m in RED.finditer(text, text.find(BS + 'begin{document}'))]
    if len(cur) != len(old):
        raise SystemExit('red passages differ: %d now, %d in the tagged version' % (len(cur), len(old)))
    out = []
    for k, (j, tags) in zip(cur, old):
        if _snippet(text, k)[:40] != _snippet(tagged, j)[:40]:
            raise SystemExit('red passages out of step:\n  now:    %s\n  tagged: %s'
                             % (_snippet(text, k), _snippet(tagged, j)))
        out.append((k, tags))
    return out


def float_tags():
    out = {}
    for m in FLOAT.finditer(_tagged()):
        lab = re.search(r'\\label\{([^}]*)\}', m.group(0))
        if lab:
            out[lab.group(1)] = _tags(m.group(0))
    return out


if __name__ == '__main__':
    text = blank_comments(open(os.path.join(HERE, '..', '..', 'paper', 'revision.tex'), encoding='utf-8').read())
    for k, tags in span_tags(text):
        print('%-22s %s' % (','.join(sorted(tags)) or '(none)', _snippet(text, k)[16:76]))
    print('--- floats')
    for lab, tags in float_tags().items():
        print('%-36s %s' % (lab, ','.join(sorted(tags)) or '(none)'))
