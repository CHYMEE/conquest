"""Build response_letter.tex from response_template.tex.

Directives in the template (each on its own line):
  %%EXCERPT{start phrase}{end phrase}  copy revision.tex from the heading of the section or
                                       subsection that contains the start phrase through the
                                       end phrase, so the revised (red) text is shown with
                                       the unchanged (black) text that leads up to it; tables,
                                       figures, and algorithms in that range are placed where
                                       they appear ([H])
  %%FLOAT{label}                       copy the table/figure environment holding \\label{label},
                                       placed here ([H]) instead of floating; skipped when an
                                       excerpt of the same location already shows it
Under each review, only the revised text that answers that review stays red; revised text that
answers other reviews is shown in black (see red_tags.py for how passages are assigned).
Within one location (\\loc), an excerpt in the same subsection as the previous one continues
from where that one stopped instead of repeating the heading.
Cross-references, equation labels, and citations inside the copied text are replaced by the
numbers they have in the compiled manuscript (read from the manuscript's .aux file).
Usage: python build_response.py <manuscript.aux>
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import red_tags  # noqa: E402
PAPER = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'paper')
aux = open(sys.argv[1], encoding='utf-8').read()
src = open(os.path.join(PAPER, 'revision.tex'), encoding='utf-8').read().split('\n')
tpl = open(os.path.join(HERE, 'response_template.tex'), encoding='utf-8').read().split('\n')

labels = {}
for m in re.finditer(r'\\newlabel\{([^}]*)\}\{\{(?:\\mbox\s*\{)?([^{}]*)\}?\}', aux):
    labels[m.group(1)] = m.group(2).strip()
cites = dict(re.findall(r'\\bibcite\{([^}]*)\}\{(\d+)\}', aux))

# the manuscript with comment lines blanked (kept as empty lines so positions stay meaningful)
TEXT = '\n'.join('' if l.lstrip().startswith('%') else l for l in src)
BODY = TEXT.find('\\begin{document}')


def resolve(s):
    def eqtag(m):
        return r'\tag{%s}' % labels[m.group(1)]
    s = re.sub(r'\\label\{(eq:[^}]*)\}', eqtag, s)
    s = re.sub(r'\\label\{[^}]*\}', '', s)
    s = re.sub(r'\\eqref\{([^}]*)\}', lambda m: '(%s)' % labels[m.group(1)], s)
    s = re.sub(r'\\ref\{([^}]*)\}', lambda m: labels[m.group(1)], s)
    s = re.sub(r'\\cite\{([^}]*)\}',
               lambda m: '{[' + ', '.join(cites[k.strip()] for k in m.group(1).split(',')) + ']}', s)
    return s


def group_end(s, i):
    """Index just past the brace group that opens at s[i] == '{'."""
    depth = 0
    k = i
    while k < len(s):
        c = s[k]
        if c == '\\':
            k += 2; continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return k + 1
        k += 1
    raise SystemExit('unbalanced braces at %d' % i)


SMALL = {'a', 'an', 'and', 'as', 'at', 'by', 'for', 'in', 'of', 'on', 'or', 'the', 'to', 'under', 'via', 'with'}


def clean_title(t):
    t = re.sub(r'\\textbf\{(.*)\}', r'\1', t.strip(), flags=re.S)
    t = ' '.join(t.split())
    if t == t.lower():
        t = ' '.join(w if (n and w in SMALL) else w[:1].upper() + w[1:] for n, w in enumerate(t.split()))
    return t


def headings():
    """(start, end of heading command, number, title) for every numbered section and subsection."""
    out, sec, sub = [], 0, 0
    for m in re.finditer(r'\\(section|subsection)\{', TEXT[BODY:]):
        a = BODY + m.start()
        b = group_end(TEXT, BODY + m.end() - 1)
        title = clean_title(TEXT[BODY + m.end():b - 1])
        if m.group(1) == 'section':
            sec, sub = sec + 1, 0
            num = roman_str(sec)
        else:
            sub += 1
            num = '%s-%s' % (roman_str(sec), chr(64 + sub))
        out.append((a, b, num, title))
    return out


def roman_str(n):
    vals = [(10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    r = ''
    for v, s in vals:
        while n >= v:
            r += s; n -= v
    return r


def roman(s):
    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50}
    n = 0
    for x, y in zip(s, s[1:] + ' '):
        n += -v[x] if y != ' ' and v[x] < v[y] else v[x]
    return n


def set_counter(label):
    num = labels[label]
    if label.startswith('tab:'):
        return r'\setcounter{table}{%d}' % (roman(num) - 1)
    if label.startswith('alg:'):
        return r'\setcounter{algorithm}{%d}' % (int(num) - 1)
    return r'\setcounter{figure}{%d}' % (int(num) - 1)


FLOAT_RE = re.compile(r'\\begin\{((?:table|figure|algorithm)\*?)\}(\[[^\]]*\])?(.*?)\\end\{\1\}', re.S)


def blacken(s):
    return s.replace('\\textcolor{red}{', '\\textcolor{black}{').replace('\\color{red}', '\\color{black}')


def recolor(start, end, review):
    """TEXT[start:end] with the red passages that do not answer this review in black.
    Floats are left as they are; here() colours each float as a whole."""
    floats = [(m.start(), m.end()) for m in FLOAT_RE.finditer(TEXT)]
    edits = []  # (position, characters replaced, new text)
    for k, tags in SPANS:
        if start <= k < end and review not in tags and not any(a <= k < b for a, b in floats):
            edits.append((k, 16, '\\textcolor{black}{'))
    # sentences inside another review's passage that also answer this review stay red
    for a, b in red_tags.extra_ranges(TEXT, review):
        if start <= a and b <= end:
            edits += [(b, 0, '}'), (a, 0, '\\textcolor{red}{')]
    s = TEXT[start:end]
    for k, n, new in sorted(edits, key=lambda e: e[0], reverse=True):
        s = s[:k - start] + new + s[k - start + n:]
    return s


def here(m, review):
    """A float environment placed in the letter at this point, numbered as in the manuscript;
    red only when it answers this review."""
    env, body = m.group(1), m.group(3)
    lab = re.search(r'\\label\{([^}]*)\}', body)
    if not lab or review not in FTAGS.get(lab.group(1), set()):
        body = blacken(body)
    base = env.rstrip('*')
    # the letter is narrower than the two-column manuscript: cap every table at the text width
    body = re.sub(r'(\\begin\{tabular\})', r'\\begin{adjustbox}{max width=\\linewidth}\n\1', body)
    body = body.replace('\\end{tabular}', '\\end{tabular}\n\\end{adjustbox}')
    lab = re.search(r'\\label\{([^}]*)\}', body)
    pre = set_counter(lab.group(1)) + '\n' if lab and lab.group(1) in labels else ''
    return pre + '\\begin{%s}[H]%s\\end{%s}' % (base, body, base)


def float_env(label, review):
    k = TEXT.find('\\label{%s}' % label)
    if k < 0:
        raise SystemExit('label not found: %r' % label)
    for m in FLOAT_RE.finditer(TEXT):
        if m.start() <= k < m.end():
            return here(m, review)
    raise SystemExit('no float holds %r' % label)


def balance(s):
    """Drop unmatched closing braces, close open braces and environments at the end."""
    out, depth, k = [], 0, 0
    while k < len(s):
        c = s[k]
        if c == '\\':
            out.append(s[k:k + 2]); k += 2; continue
        if c == '{':
            depth += 1
        elif c == '}':
            if depth == 0:
                k += 1; continue
            depth -= 1
        out.append(c); k += 1
    s = ''.join(out) + '}' * depth
    stack = []
    for m in re.finditer(r'\\(begin|end)\{([^}]*)\}', s):
        if m.group(1) == 'begin':
            stack.append(m.group(2))
        elif stack and stack[-1] == m.group(2):
            stack.pop()
    return s + ''.join('\n\\end{%s}' % e for e in reversed(stack))


def strip_comments(s):
    return '\n'.join(re.sub(r'(?<!\\)%.*$', '', l) for l in s.split('\n'))


HEADS = headings()
SPANS = red_tags.span_tags(TEXT)  # (position, comments answered) of every red passage
FTAGS = red_tags.float_tags()     # comments answered by each table, figure, and algorithm


def excerpt(a, b, prev, review):
    """(latex, (heading start, end position), float labels shown) for one EXCERPT directive."""
    i, j = locate(a, b)
    end = j + len(b)
    h = [x for x in HEADS if x[0] <= i][-1]
    if prev and prev[0] == h[0]:
        if end <= prev[1]:
            return '', prev, set()
        start, head = prev[1], ''
    else:
        start, head = h[1], '\\mshead{%s}{%s}\n' % (h[2], h[3])
    s = strip_comments(recolor(start, end, review))
    shown = set(re.findall(r'\\label\{([^}]*)\}', ''.join(m.group(0) for m in FLOAT_RE.finditer(s))))
    s = FLOAT_RE.sub(lambda m: here(m, review), s)
    s = re.sub(r'\n{3,}', '\n\n', s).strip('\n')
    return head + balance(s), (h[0], end), shown


def locate(a, b):
    """Positions of the start and end phrases: the closest pair with the start before the end,
    so a short start phrase such as '\\textcolor{red}{The' still finds its own passage."""
    best = None
    i = TEXT.find(a, BODY)
    while i >= 0:
        j = TEXT.find(b, i + len(a))
        if j < 0:
            break
        if best is None or j - i < best[1] - best[0]:
            best = (i, j)
        i = TEXT.find(a, i + 1)
    if best is None:
        raise SystemExit('phrases not found in order: %r ... %r' % (a, b))
    return best


def split_phrases(body):
    """Split 'start}{end' at the '}{' giving the longest start phrase that occurs before the end."""
    for k in reversed([m.start() for m in re.finditer(r'\}\{', body)]):
        a, b = body[:k], body[k + 2:]
        i = TEXT.find(a, BODY)
        if i >= 0 and TEXT.find(b, i + len(a)) >= 0:
            return a, b
    raise SystemExit('cannot split directive: %r' % body)


# blocks: a location (\loc) or review starts a new block; excerpts in a block share context
blocks, cur = [], []
for line in tpl:
    if re.match(r'\s*\\(loc|review)\b', line) and cur:
        blocks.append(cur); cur = []
    cur.append(line)
blocks.append(cur)

out, review = [], None
for block in blocks:
    prev, shown, made = None, set(), {}
    r = re.match(r'\s*\\review\{(\d\.\d+)\}', block[0])
    if r:
        review = 'R' + r.group(1)
    for n, line in enumerate(block):
        m = re.match(r'\s*%%EXCERPT\{(.*)\}\s*$', line)
        if m:
            tex, prev, sh = excerpt(*split_phrases(m.group(1)), prev, review)
            shown |= sh
            made[n] = tex
    for n, line in enumerate(block):
        f = re.match(r'\s*%%FLOAT\{(.*)\}\s*$', line)
        if n in made:
            if made[n]:
                out += [r'\begin{excerpt}', resolve(made[n]), r'\end{excerpt}']
        elif f:
            if f.group(1) not in shown:
                out.append(resolve(float_env(f.group(1), review)))
                shown.add(f.group(1))
        else:
            out.append(line)
open(os.path.join(PAPER, 'response_letter.tex'), 'w', encoding='utf-8', newline='').write('\n'.join(out))
print('written response_letter.tex')
