# Breast DCE-MRI Reliability — interactive demo

Frontend-only React + TypeScript + Vite prototype for the research reproducibility companion.

This is intentionally a **demo interface**. It does not execute nnU-Net or any clinical workflow. The animated pipeline and terminal replay the already verified analysis steps and numbers from the public repository.

## What is functional

- animated six-stage research pipeline
- typewriter-style execution log
- replay button
- running / verified state and timer
- representative MRI cases using the public manuscript panels
- real case IDs and real Raw/LCC Dice values for the two displayed media examples
- click-to-open MRI viewer
- image zoom controls
- external cohort summary
- manuscript-to-code map
- direct repository link

## Source values used in the demo

Representative case:
- I-SPY1 / `ISPY1_1128`
- Raw Dice: 0.381484...
- LCC Dice: 0.816927...

LCC-collapse case:
- Duke / `DUKE_883`
- Raw Dice: 0.522466...
- LCC Dice: 0.000

The case selection and values come from:

```text
12_media_orthogonal_figures/selected_cases_for_media_figure.csv
```

The displayed images are loaded from the corresponding public files already committed in `MRI-Project`.

## Run locally

From the repository root:

```bash
cd reproducibility_site
npm install
npm run dev
```

Vite will print a local address, normally:

```text
http://localhost:5173/
```

Open it in your browser.

## Production build

```bash
npm run build
npm run preview
```

The production files are written to:

```text
reproducibility_site/dist/
```

## No backend is required yet

The current demo is deliberately frontend-only. A backend becomes useful later if you want to:

- launch real Python scripts from the browser
- stream real subprocess logs instead of replaying a verified demo trace
- browse new locally uploaded MRI/NIfTI studies
- store run histories
- expose generated artifacts dynamically

For the portfolio/reviewer demo, keeping this first version static makes it easier to run and safer to publish.

## Later GitHub Pages deployment

The current Vite config uses relative built asset paths (`base: "./"`), so the build is portable. When we are ready to publish the site, we can add a small GitHub Pages workflow or deploy the `dist` directory separately.
