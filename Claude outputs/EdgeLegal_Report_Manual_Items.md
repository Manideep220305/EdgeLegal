# EdgeLegal Report: Items That Cannot Be Generated Automatically

> Historical snapshot. Use [current research progress](../docs/EdgeLegal_Progress.md) for implementation and experiment status.

This file belongs with `EdgeLegal_Report_Context.md`. It lists everything the Word report needs that Claude cannot produce on its own: things that need your real application running, your personal details, results that do not exist yet, or decisions only you can make. For each item it gives what to do and the text or description that goes in the report meanwhile. In the Word document each of these will appear as a clearly marked placeholder box (caption plus a one-line description) so you can drop the real item in.

This file and the Word report stay out of the project folder.

---

## A. Identity and administrative pages (you must provide)

| Item | Where | What is needed | Placeholder shown in the report |
|---|---|---|---|
| Team members and roll numbers | Title page, Certificate, Declaration, Acknowledgement | Full names and roll numbers of every member | `[Student name]  [Roll No.]` rows |
| Internal guide name and designation | Title page, Certificate, Acknowledgement | Name, designation | `[Guide name], [Designation]` |
| Head of Department | Certificate | Name and designation | `[HoD name], Professor and HoD` |
| External examiner line | Certificate | Left blank for signature | blank signature line |
| Institution name, address, logo image | Title page, Certificate | Confirm which institution appears in the final report and provide the logo file (PNG or JPG, at least 300 px wide). The Review-II deck and the reference report are from GRIET; your profile mentions BVRIT. Tell Claude which one to use. | `[Institution logo]` box and `[Institution name]` |
| Academic year, date, place | Title page, Declaration | Confirm (draft uses 2026-2027) | `2026-2027`, `[Date]`, `[Place]` |
| Batch number | Title page | Review-II used Batch No. C7 | `Batch No: C7` (confirm) |
| Signatures | Certificate, Declaration | Hand-signed after printing | blank lines |

Acknowledgement draft text (edit names):

> We express our sincere gratitude to our internal guide, [Guide name], [Designation], Department of Computer Science and Engineering, for the guidance, encouragement and constructive criticism received throughout this project. We thank [HoD name], Professor and Head of the Department, for the facilities and support provided, and the project review committee for its valuable suggestions during the reviews. We are grateful to the faculty and staff of the department and to our families and friends for their support. We also acknowledge the authors of the open-source datasets, models and tools on which this work builds.

---

## B. Screenshots (need your running application)

Run the app (`cd backend && uvicorn app.main:app --reload`, `cd frontend && npm run dev`), use the demo contract in `pdfs/`, and capture at 1600 x 900 or larger. PNG, light or dark theme consistently. Each item below becomes a figure in Section 5.4 or Appendix II.

| Fig | Caption | What to capture | Fallback description shown if missing |
|---|---|---|---|
| 5.4.1 | Landing Page | Top of the landing page | "Landing page introducing EdgeLegal with a call to action to analyze a contract." |
| 5.4.2 | Upload Panel with a Contract Loaded | /app page after uploading the demo contract, PDF visible in the viewer | "Analyze page showing the uploaded contract in the PDF viewer and the hypothesis input." |
| 5.4.3 | Hypothesis Input and Analysis in Progress | The step rail and the spinning indicator on the hypothesis being checked | "Progress state while a hypothesis is being analyzed." |
| 5.4.4 | Verdict Resolved by the Classifier (Accept Path) | A result with engine "classifier", confidence bar, cited clause; decision trail showing accept | "Result card with label, confidence and the decision trail showing the classifier accepted the case." |
| 5.4.5 | Verdict Escalated to the LLM | A result with engine "LLM", reasoning text, decision trail showing escalate. (Needs a confidence below 0.95 or a temporarily higher threshold in a test.) | "Result card showing escalation to the LLM with reasoning and citations." |
| 5.4.6 | Citation Jump in the PDF Viewer | Clicking a citation scrolls the viewer to the cited page | "PDF viewer positioned on the page of the cited clause." |
| 5.4.7 | History Page | History list with filters and search | "History page listing saved analyses with label, confidence, engine and time." |
| 5.4.8 | Analytics Dashboard | KPI tiles and the most contradicted hypotheses chart | "Analytics page with usage totals, contradiction rate and classifier versus LLM split." |
| 5.4.9 | Research Page: Dev Metrics and Reliability Diagram | Stat tiles, reliability diagram, threshold sweep | "Research page showing dev-split accuracy, macro-F1, ECE, reliability diagram and threshold sweep." |
| 5.4.10 | Research Page: A/B/C Comparison | The config comparison section with its warning banners | "Research page comparison of configurations with banners for non-paper-grade runs." |
| A.1 to A.8 | Appendix II (Input, Process, Output) | Upload (input), hypothesis entry (input), indexing and analysis in progress (process), decision trail (process), result (output), history (output), analytics (output), research (output). Reuse the above where they fit. | Same captions with "(input)", "(process)" or "(output)". |

Optional extras: a terminal screenshot of the experiment runner printing its comparison table, and a screenshot of `logs/edgelegal.log` showing routing decisions (retrieve, triage, confidence_check, accept or escalate).

---

## C. Measurements and results that do not exist yet

| Item | Status | What to do | What the report says meanwhile |
|---|---|---|---|
| Matched A/B/C run (same 20 documents, 340 pairs, all three configs) | Not completed (stopped at the daily Groq token cap) | From the repository root with the backend environment: `python -m backend.app.experiment_runner --configs A B C --limit 20` (use `--limit 10` if the cap is hit). Send Claude the three new files in `results/`. | Table 5.7 rows for A, B and C marked "pending"; text states the run is queued and Groq numbers are not paper-grade. |
| McNemar significance between configs | Needs the matched run | Computed from the result files by Claude once they exist | "Not yet computed." |
| Paper-grade run (full 123 documents, local LLM, repeats) | Not started (deliberately last) | Needs local Ollama, retrained classifier, improved retrieval | Described only in Future Scope. |
| Retrained classifier (realistic premises) and re-run of the in-pipeline check | Not started | Retrain, then re-run the in-pipeline triage check and `retrieval_accuracy_check.py` | Future Scope; current numbers reported as the negative finding. |
| Automated test suite results | No checked-in suite exists | Claude will run the checks it can (history database, routing logic, API with stubbed model) and record actual output. Anything it cannot run is listed here as not run. | Section 5.1 and 5.2 contain only executed checks, dated. |
| API response-time table (like the reference's Table 5.7) | Not measured | Run the timing script below against a running backend and send the output | Table 5.8 omitted or marked "not measured"; sentence stating latency with the local LLM has not been measured. |
| Frame or CPU usage, local-LLM latency | Not measured | Needs the local LLM | Stated as not measured. |
| A formal user study | Not carried out | Optional future work | Listed in limitations. |

Timing script (run from the project root with the backend running; prints median, mean, 95th percentile and max in milliseconds over 20 calls per endpoint):

```python
import statistics, time, urllib.request, json

BASE = "http://localhost:8000"
ENDPOINTS = ["/health", "/history?limit=20", "/history/documents", "/analytics", "/research/metrics"]

def timeit(path, n=20):
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        urllib.request.urlopen(BASE + path).read()
        xs.append((time.perf_counter() - t) * 1000)
    xs.sort()
    return statistics.median(xs), statistics.mean(xs), xs[int(0.95 * (n - 1))], xs[-1]

for p in ENDPOINTS:
    med, mean, p95, mx = timeit(p)
    print(f"{p:25s} median {med:6.1f}  mean {mean:6.1f}  p95 {p95:6.1f}  max {mx:6.1f} ms")
```

---

## D. Facts that need a human check before submission

1. **References.** Claude supplies a candidate list from the project's literature notes. Check each entry's authors, venue, volume and pages against the original source before final submission. Two need extra care: Better Call CLAUSE (author list not recorded in the notes) and the BGE embedding paper. The ACM ICBAR 2025 paper "Multi-Agent Orchestration of Local LLMs for Contract Review under Resource Constraints" (DOI 10.1145/3800227.3800266) could not be read; it is mentioned in the survey as inaccessible and should be read through your library before you finalize the survey.
2. **Software versions** in Table 3.3.4 come from `backend/requirements.txt` and `frontend/package.json` at build time. Confirm they match your machines.
3. **Hardware table.** Claude lists development laptops as CPU-only with 12 to 16 GB RAM from project notes. Confirm the actual specifications of the machine used for results.
4. **Training details** (epochs, batch size, learning rate, Kaggle T4) are from the project notes and the checkpoint's training script. Confirm with the teammate who ran the Kaggle training.
5. **Deviations from Review-II** (DeBERTa-v3-base instead of LegalBERT, Groq instead of local Llama or Phi-3 for now, no report generator or multi-turn Q&A) are stated in the report. Be ready to explain them in the viva.
6. **Privacy wording.** The report says the design goal is local processing and that the interim Groq path sends escalated text to a hosted service. Do not change this to an unconditional privacy claim.

---

## E. Text blocks that stand in for things the Word file cannot contain

- **Plagiarism or AI-use declaration:** some departments require a similarity report or a statement about tool use. If yours does, attach the official report at the end; Claude cannot generate it.
- **Live demo:** the report cannot embed a video. Mention in Section 5.4 that a recorded demonstration is available on request, if you record one (a 2 to 3 minute screen recording of upload, analysis, history and the Research page is enough).
- **Interactive charts:** the Research page charts are interactive; the report contains static equivalents generated from the same data files.
- **Colour:** if the department prints in black and white, tell Claude so the diagrams use patterns instead of colour.

---

## F. Checklist before you ask for the Word document

1. Tell Claude the institution (logo and name), team members, guide and HoD, batch number, and the date.
2. Provide the screenshots (section B), even a partial set; missing ones become labelled boxes.
3. Run the matched A/B/C experiment (section C) if you want real numbers in Table 5.7; otherwise it stays "pending".
4. Optionally run the timing script and send the output.
5. Say whether the report should be black-and-white print friendly.
