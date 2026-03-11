PYTHON ?= python

.PHONY: prepare smoke baselines main ablation reviewer summarize tables figures report all

prepare:
	$(PYTHON) scripts/prepare_all.py --config configs/paths.yaml

smoke:
	$(PYTHON) scripts/run_suite.py --suite configs/suite/smoke.yaml

baselines:
	$(PYTHON) scripts/run_suite.py --suite configs/suite/baselines.yaml

main:
	$(PYTHON) scripts/run_suite.py --suite configs/suite/main.yaml

ablation:
	$(PYTHON) scripts/run_suite.py --suite configs/suite/ablation_all_models.yaml

reviewer:
	$(PYTHON) scripts/run_suite.py --suite configs/suite/reviewer.yaml

summarize:
	$(PYTHON) scripts/summarize_results.py

tables:
	$(PYTHON) scripts/export_tables.py

figures:
	$(PYTHON) scripts/export_figures.py

report:
	$(PYTHON) scripts/package_report.py

all:
	$(PYTHON) scripts/run_pipeline.py --suite configs/suite/all.yaml

