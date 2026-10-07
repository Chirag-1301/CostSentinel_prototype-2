CostSentinel prototype - Team 27
================================

IN PLAIN ENGLISH (read this first)
----------------------------------
What is CostSentinel?
  A tool that watches the costs of every job in a factory each day and warns the
  plant controller when a job looks like it will finish more than 10% over budget,
  while there is still time to do something about it. Today most factories only
  see this in the monthly cost report, often after the job has already finished.

What is in this folder?
  1. CostSentinel_Dashboard.html  - the demo. Double-click to open it in any browser
     (no internet or installation needed). Pick a job, press "Replay job", and watch
     when CostSentinel would have warned you compared with the monthly report.
  2. costsentinel_prototype.py    - the engine. It (a) creates two years of made-up
     factory cost data, (b) teaches an AI model what normal spending looks like using
     year 1, and (c) tests it on year 2 against the monthly report and two simple rules.
  3. results/                     - every number used in our deck, as CSV/JSON files.

Is the data real?
  No. All data is simulated (made up by our code) because we had no access to a real
  factory's cost records. The simulation lets us test HOW the idea works and which
  data problems break it. It does not prove real-world savings. The next step is to
  run the same code on one real factory's past data, then a controlled pilot.

How do I re-create the results?
  Install Python 3.13, then run:   python -m pip install -r requirements.txt
                                   python run_all.py
  This rebuilds every result file and the dashboard (takes several minutes).

Words used in the files
  Overrun job          - a job that finishes more than 10% over its budget.
  Caught in time       - the first real warning came before the job finished
                         (technical name: open-job recall).
  Warnings that are real - share of all warnings that pointed at a real problem
                         (technical name: precision).
  Extra cost still ahead - share of the problem's extra cost that had not yet been
                         spent when the warning came; the part you could still save.
  Risk score           - a 0-1 ranking of how unusual a job looks; NOT a % chance.
  Base case            - our central scenario with some messy data (15% of costs
                         entered late, 30% of jobs with weekly timesheets, 3% booked
                         on the wrong job). Called "Typical" inside the code.

================================================================================
TECHNICAL DETAILS (unchanged from the team's documentation)
================================================================================
CostSentinel prototype — Team 27

OPEN THE DEMO
Open CostSentinel_Dashboard.html in a browser. The page is an offline replay of
13 selected synthetic jobs. It is not connected to an ERP or a running model.
Explanations are deterministic templates. Verdict buttons demonstrate a proposed
workflow; the dashboard states exactly how feedback is stored. Nothing is sent
to an ERP, email service or external language model.

REPRODUCE EVERY RESULT AND REBUILD THE DASHBOARD
Use Python 3.13 with the pinned versions in requirements.txt. The validated run
used Python 3.13.5, NumPy 2.1.3, pandas 2.2.3, SciPy 1.15.3 and scikit-learn 1.6.1.
No accounts, API keys or downloaded data are needed.

    python -m pip install -r requirements.txt
    python run_all.py

The entrypoint runs all four data-quality scenarios, all three single-factor
ablations, the test-set threshold diagnostic, driver/failure analysis and sample
export, then rebuilds the dashboard from results/dashboard_data.json. It uses one
compute thread. Allow several minutes; the exact time depends on the computer.
All generated files remain inside this folder. The HTML opens without Python.

FILES
costsentinel_prototype.py    Simulation, classifier, baselines and evaluation.
ablation.py                 Optional standalone rerun of the three ablations.
run_all.py                  Single end-to-end reproduction entrypoint.
build_dashboard.py          Builds HTML from the current dashboard data.
requirements.txt            Exact package versions used for verification.
results/                    CSV/strict-JSON outputs, including metric definitions.
results/validation_threshold_curves.csv records every tested calibration choice.
results/typical_threshold_curve.csv is a TEST diagnostic; it never selects a threshold.

WHAT WAS TESTED
One synthetic plant, three job families, 1,200 generated jobs per annual cohort.
This is a test of a proposed mechanism, not evidence of real-world
accuracy, user adoption, willingness to pay or realized savings.
Data-quality percentages are nominal probabilities. A mis-coding attempt can
select the same active job, so the realized wrong-job share is slightly lower.


INPUT PROVENANCE
Job counts, family mix, budget/duration ranges, cost patterns, cause frequencies
and posting-error rates are modelling assumptions, not parameters calibrated to
industry ERP data. External sources in the deck support business context; they
do not supply these simulated inputs. For example, APQC's close-cycle benchmark
does not establish the assumed five-working-day month-end report lag. Detection
metrics are calculated simulation outputs; financial value also depends on the
stated effectiveness, review-cost and commercial assumptions.

The dashboard and deck call the illustrative central scenario the "Base case"
or "assumed base case". Its machine-readable scenario key is "Typical" in the
source code and results. That key does not mean the assumed posting-error rates
are empirically typical of manufacturers.

Training uses only jobs closed by absolute working day 252 and postings available
before that cutoff. The test cohort starts on day 252. Posting degradation is
performed separately by cohort, so mis-coding cannot transfer a test posting into
training. A job-level 25% validation split chooses alert thresholds; learned cost
curves are fitted on the model-training fold before validation. Curves and the
classifier are then refitted on all eligible history without test observations.

All real-time arms use DAILY calibration and test snapshots, starting at day 1,
and two consecutive daily threshold exceedances. The two simple rules and ML use
the same validation objective: maximize remaining simulated driver cost while at
least 50% of emitted alerts are valid. Candidate thresholds: ML score 0.20–0.95;
linear and phased cost ratios 1.05–2.50, both in steps of 0.05. If no candidate
passes, the highest-precision candidate is retained for diagnostic comparison
only and explicitly marked infeasible. Held-out pass/fail is reported separately.
No threshold is changed in response to test results.

All arms use the same five-working-day cooldown after a dismissed alert. Monthly
reports keep their fixed 10% phased-budget variance rule and five-day report lag;
they are a stated status-quo comparator, not a tuned real-time alternative.

MODEL SCORE
The training target is: observed final cost >110% of budget AND current excess
above the learned family cost curve >3% of total budget. The displayed classifier
score is not a calibrated probability that a job will finish over budget. The
simple completion projection is budget × cost-to-date / planned-cost-to-date,
not a separately validated forecast model. Explanations identify observed cost
elements and possible checks; they do not establish a causal diagnosis.

METRICS AND LIMITATIONS
The simulated manager has oracle information: an alert is valid only if the job
truly ends >10% over budget and its injected cause has already started. All other
alerts are dismissed; the first valid catch ends monitoring for that job. Real
managers do not have this knowledge. No human-response experiment has occurred.

Precision = valid catches / emitted alerts, including repeated dismissed alerts.
Open-job recall = valid catches before planned closure / all true-overrun jobs.
After-close share = after-close valid catches / all valid catches, NOT all alerts.
Weekly load = cohort alerts / 50.4 working weeks; review cost uses the exact alert
count, 30 minutes per alert and EUR60 per hour.

Remaining driver cost = injected overrun-driver cost on true-overrun jobs that is
scheduled at or after a valid alert. Its denominator is injected driver cost, not
total overspend. The main metric assumes zero response delay; a two-working-day
response sensitivity is also exported. Remaining costs may already be committed
or unavoidable. Applying 15%, 30% or 45% effectiveness gives conditional value
scenarios, not measured savings. A job being open does not prove it is fixable.
comparator_value.csv/.json apply the same 30% effectiveness assumption to all arms.
The phased rule can have greater conditional euro value while failing the alert
precision constraint. Rule setup/maintenance costs are unestimated; no rule
subscription is assumed. The table compares standalone-arm opportunities, not
measured savings or a simulation of combined monthly-plus-real-time workflows.

The monthly comparator uses only postings received before month-end and excludes
accruals and arrivals during the close. Other finance practices may perform better.
The model is tested on one stationary synthetic data-generating process and one
seed. No uncertainty interval or field validation is claimed. Simulated precision
checks, whether passed or failed, never constitute production approval.

REAL-DATA NEXT STEP
Define an ERP data contract before a real backtest: job IDs, planned start and
duration, family, element budgets, posting dates/amounts, final costs and reliable
correction history. A real backtest cannot assume known cause onset or injected
extra-cost ground truth; its labels and metrics must be redesigned accordingly.
Persistent integrations, audit trails, access controls, data-readiness checks,
posting-lag monitoring and feedback-driven retraining are proposed product work.
