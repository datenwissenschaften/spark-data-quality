.PHONY: notebooks

# Re-executes the end-to-end notebook top to bottom, regenerating its embedded
# outputs and the Plotly PNG assets under docs/assets/data-quality-workflow/.
# Fails on any notebook cell error. Requires the `fugue` and `notebook` extras:
#   poetry install --with dev --extras "fugue notebook"
notebooks:
	poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/data_quality_workflow.ipynb
