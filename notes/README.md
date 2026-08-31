# notes — theory notes for `rotational-diffusion-photophysics`

Source files for the theory/simulation notes. **Everything here is source; nothing here
is generated.** All build output goes to `build/`, which is git-ignored — never edit a
file in `build/`, and never commit one.

| file | what it is |
|---|---|
| `notes.tex` | the current document (LaTeX; STARSS methods 1–3 + the 2022 S² formalism) |
| `formalism.md` | **draft** replacement for the theory chapter, in markdown: the representation-agnostic core, the S² representation, and the photophysical schemes. Not yet spliced into `notes.tex` |
| `references.bib` | bibliography, used by both the LaTeX and the markdown routes |
| `figures/` | figure sources (`.png` for inclusion, `.ai`/`.pptx` originals) |
| `header.tex` | layout preamble for the markdown route (packages only — **not** biblatex) |
| `build/` | **generated** — PDFs, `.tex` from markdown, and all LaTeX aux files. Git-ignored |

New chapters are written in **markdown** and converted to LaTeX/PDF with pandoc; the
legacy `notes.tex` stays LaTeX. Rationale, the audit of what needs rewriting, and the
full authoring conventions live in the research repo:
`studies/a00/a60_notes_update/n010_notes_extension_plan.md`.

## Tooling

- **pandoc 3.10.2** (`winget install JohnMacFarlane.Pandoc`) — on PATH at
  `%LOCALAPPDATA%\Pandoc`. Note Quarto also ships an older pandoc (2.19.2); if a build
  ever behaves oddly, check which one ran with `where pandoc`.
- **TinyTeX** — `pdflatex`, `biber`, `xelatex`, `latexmk`, `tlmgr`. Currently unpacked
  at `%USERPROFILE%\Downloads\TinyTeX\bin\windows` and **not on PATH**; either add that
  directory to PATH or prefix the commands below with it. Missing LaTeX packages are
  installed on demand with `tlmgr install <package>` (this document needed `tocloft`
  and `biblatex-phys` on a fresh TinyTeX).

## Build the LaTeX document

From this directory:

```powershell
mkdir build -Force
pdflatex -interaction=nonstopmode -output-directory=build notes.tex
biber --input-directory build --output-directory build notes
pdflatex -interaction=nonstopmode -output-directory=build notes.tex
pdflatex -interaction=nonstopmode -output-directory=build notes.tex
```

Result: `build/notes.pdf`. Two `pdflatex` passes after `biber` are needed to settle the
table of contents, list of figures, and citation numbers.

Expected output as of 2026-08-26: 27 pages, 0 errors, and exactly **2** undefined
references (`sec:high_power_offswitching`, `sec:starss2_fit` — they point into the
parent SI document and are known). Anything else is a regression.

## Build a markdown chapter

```powershell
pandoc chapter.md -f markdown -t latex --standalone -N `
  --biblatex --bibliography=references.bib `
  -V biblatexoptions="style=phys,articletitle=false,biblabel=brackets,chaptertitle=false,pageranges=false,sorting=none" `
  -H header.tex -o build/chapter.tex
pdflatex -interaction=nonstopmode -output-directory=build build/chapter.tex
biber --input-directory build --output-directory build chapter
pdflatex -interaction=nonstopmode -output-directory=build build/chapter.tex
pdflatex -interaction=nonstopmode -output-directory=build build/chapter.tex
```

`header.tex` holds the layout packages only (`mathtools`, `authblk`, `tocloft`, …).
**biblatex must not be loaded there** — it comes from pandoc's template via `--biblatex`
plus `-V biblatexoptions`, and loading it in both places causes an option clash.

### Markdown authoring conventions

Write plain pandoc markdown, with two exceptions where raw LaTeX goes straight into the
`.md` file (pandoc passes it through untouched):

- **Numbered equations** — use the raw environment and reference it with `\eqref`:

  ```latex
  \begin{equation}\label{eq:master}
      \frac{\partial p}{\partial t} = D_R \nabla^2 p + \sum_\eta k\, p
  \end{equation}
  ```

  Numbering stays LaTeX's, so it renumbers correctly. Use `$$…$$` where no number is
  wanted, and `$…$` inline.
- **Figures needing a short caption** (anything that must read well in the list of
  figures) — use the raw `figure` block with `\caption[short]{long}`. Plain
  `![caption](figures/x.png){width=80%}` is fine for figures that don't need one.

Citations are ordinary pandoc syntax, `[@Khatib2016]`, resolved by biber against
`references.bib` in the existing `phys` style. `\tableofcontents`, `\listoffigures` and
`\printbibliography` also go in as raw blocks.

## Known defects (see the a60 plan)

- `references.bib` defines `Yadav2015` and `Masullo2018` **twice**; biber silently skips
  the second of each pair.
- `notes.tex` §1.4 documents the Wigner-3j spherical-harmonic product rule, which is
  **not** what the engine implements (it uses a real-SH product built by quadrature).
  Correcting it is the first writing task.
