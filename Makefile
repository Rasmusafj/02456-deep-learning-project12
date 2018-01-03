TEST_PATH=./tests

init:
	conda env create -f environment.yml

update:
	conda env update -f environment.yml

test:
	PYTHONPATH=. py.test --verbose --color=yes $(TEST_PATH)

.PHONY: init test
