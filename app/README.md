# Dashboard application

> **Warning:** This dashboard is a student Final Year Project prototype for academic demonstration only. It is not a medical device, has not been clinically validated, and must not be used for diagnosis or medical decision-making. Users operate it at their own risk. The student authors, supervisors, university, and contributors accept no responsibility or liability for its use or misuse.

The Gradio dashboard is provided in `app.py`. By default it looks for checkpoints and class metadata in the repository's `models/` directory. To use another location, set `FYDP_PROJECT_DIR`; to change the report directory, set `FYDP_APP_DIR`.

Model files are intentionally omitted from this repository; the dashboard will display a model-not-found message until a checkpoint is placed in the expected location.

The final application instructions should explain:

```bash
python app.py
```

Install the dependencies from the repository root before launching the dashboard. The dashboard is for research demonstration and is not a medical diagnostic tool.

Example with an external model directory:

```bash
FYDP_PROJECT_DIR=/path/to/models python app.py
```
