# Build bigdata4eda.pdf from bigdata4eda.md. Run from paper/.
#
# Flag rationale (md-to-latex skill), so the knowledge is not tribal:
#  - Title/author/abstract live in the .md YAML front matter, so the body
#    headings start at `##`; `--shift-heading-level-by=-1` maps them to
#    \section (otherwise pandoc emits no \section and numbering degrades).
#  - `-F pandoc-crossref` resolves the {#sec:..}/{#eq:..}/{#tbl:..} labels
#    and the @sec:/@eq: references in the prose.
#  - `--citeproc` builds the reference list from bigdata4eda.bib using the
#    vendored, self-contained ieee.csl (no independent-parent): offline and
#    reproducible.
#  - Build options live in paper/latex.yaml (IEEEtran journal class,
#    hyperlinks) and paper/crossref.yaml (reference prefixes); they are
#    loaded with --metadata-file because pandoc 3.11 does not read a
#    positional *.yaml as metadata.
#  - Two-column IEEEtran: pandoc emits `longtable` for every pipe table,
#    which is illegal in two-column mode, so the tables and figures are
#    authored as raw LaTeX `table*`/`figure*` floats; their \label s are
#    plain LaTeX labels referenced with Table~\ref{...}/Figure~\ref{...}.
#  - Figures are PNG and pdflatex embeds them directly; no converter step.
#
# Requires: pandoc, pandoc-crossref, a LaTeX engine with pdflatex.

PANDOC := pandoc
FLAGS  := -F pandoc-crossref --citeproc -s -t latex -N --reference-links \
          --shift-heading-level-by=-1 --pdf-engine=pdflatex \
          --metadata-file=latex.yaml --metadata-file=crossref.yaml

FIG_PNG := $(wildcard figures/*.png)

.PHONY: paper slides clean

paper: bigdata4eda.pdf

bigdata4eda.pdf: bigdata4eda.md bigdata4eda.bib ieee.csl latex.yaml \
                 crossref.yaml $(FIG_PNG)
	$(PANDOC) $(FLAGS) bigdata4eda.md -o $@

# --- Beamer slides ---------------------------------------------------------
# The deck needs LuaLaTeX (emoji font fallback) and two passes (outline and
# section navigation), so emit slides.tex with pandoc, then run lualatex twice.
# The preamble is a CLI flag: pandoc mangles raw LaTeX in YAML header-includes.
slides: bigdata4eda-slides.pdf

bigdata4eda-slides.tex: bigdata4eda-slides.md bigdata4eda-beamer.yaml \
                        bigdata4eda-preamble.tex bigdata4eda.bib ieee.csl
	$(PANDOC) --citeproc -s -t beamer --slide-level=2 \
	  --metadata-file=bigdata4eda-beamer.yaml \
	  --include-in-header=bigdata4eda-preamble.tex \
	  bigdata4eda-slides.md -o $@

bigdata4eda-slides.pdf: bigdata4eda-slides.tex $(FIG_PNG)
	lualatex -interaction=nonstopmode -halt-on-error bigdata4eda-slides.tex
	lualatex -interaction=nonstopmode -halt-on-error bigdata4eda-slides.tex

clean:
	-@rm -f bigdata4eda.tex bigdata4eda.aux bigdata4eda.log \
	       bigdata4eda.out bigdata4eda.toc \
	       bigdata4eda-slides.aux bigdata4eda-slides.log \
	       bigdata4eda-slides.out bigdata4eda-slides.nav \
	       bigdata4eda-slides.snm bigdata4eda-slides.vrb ; true
