# crypto-microstructure: record, fetch, verify, measure.
#
#   make test                     run the test suite
#   make fetch DAY=2025-08-01     pull one Tardis free day (DD=01, no key)
#   make verify                   re-hash every file the manifest lists
#   make results                  regenerate results/ from manifest.json
#   make smoke                    run the whole pipeline over the synthetic
#                                 fixture into results/self-test/
#   make fixture                  rebuild the committed synthetic fixture
#   make record                   start the recorder (Ctrl-C to stop)
#   make hash                     add recorded files to the manifest
#   make clean                    remove caches and generated smoke output
#
# PYTHON defaults to the repo venv when it exists, else `python`.

PYTHON ?= $(shell test -x .venv/Scripts/python.exe && echo .venv/Scripts/python.exe \
                  || (test -x .venv/bin/python && echo .venv/bin/python) \
                  || echo python)

MANIFEST ?= manifest.json
OUT      ?= results
DATA     ?= data
SYMBOL   ?= BTC-USD
EXCHANGE ?= coinbase
FOLDS    ?= 4
NBOOT    ?= 200
MINUTES  ?= 1440

FIXTURE_MANIFEST := tests/fixtures/synthetic/manifest.json

.PHONY: all test fixture results smoke fetch verify hash record clean help

all: test results

help:
	@sed -n '2,12p' Makefile

test:
	$(PYTHON) -m pytest -q tests

# One Tardis day. Without an API key only the first of a month is served,
# which is why DAY must end in -01 unless TARDIS_API_KEY is set.
#   make fetch DAY=2025-08-01
#   make fetch DAY=2025-08-01 MINUTES=10        # one-request smoke
fetch:
	@test -n "$(DAY)" || (echo "usage: make fetch DAY=YYYY-MM-01 [MINUTES=1440]"; exit 2)
	$(PYTHON) tardis_loader.py fetch --day $(DAY) --exchange $(EXCHANGE) \
		--symbol $(SYMBOL) --minutes $(MINUTES) --out $(DATA) --manifest $(MANIFEST)

# sha256 every file the manifest lists. Missing files are reported but do
# not fail; a CHANGED file does, because a published number would no longer
# correspond to the bytes on disk.
verify:
	$(PYTHON) tardis_loader.py verify --manifest $(MANIFEST)

hash:
	$(PYTHON) tardis_loader.py hash --out $(DATA) --manifest $(MANIFEST)

record:
	$(PYTHON) record.py

# Every table in results/ comes out of this one command. With an empty
# manifest it still runs and the report says, in writing, that no day is
# analysable yet -- that is the honest state until the owner pulls days.
results:
	$(PYTHON) analyze.py results --manifest $(MANIFEST) --out $(OUT) \
		--folds $(FOLDS) --n-boot $(NBOOT)

# The same pipeline over the committed synthetic fixture: no network, about
# a second, and it produces a real (tiny) table. Not a market result.
smoke:
	$(PYTHON) analyze.py results --manifest $(FIXTURE_MANIFEST) \
		--out $(OUT)/self-test --folds 3 --n-boot 50 \
		--horizons 1,5,10,30,60,300

fixture:
	$(PYTHON) tests/fixtures/make_fixture.py

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ tests/__pycache__ ci.log
