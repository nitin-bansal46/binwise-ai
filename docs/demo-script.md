# Three-minute demo script

## 0:00–0:30 — problem and boundary

“BinWise answers a deceptively local question: where does this item go? The classifier recognises
nine broad materials, but the disposal instruction comes from a city-filtered structured rule. The
model is not allowed to invent municipal policy.”

Show the architecture diagram and mention the two target cities/languages.

## 0:30–1:15 — normal case

Upload a clear glass-container image, select Zurich and German, and analyse. Point out confidence,
preparation, exceptions, official source, verification date and the machine-translation notice.
Open the trace and show the direct path from classification to retrieval and grounding validation.

## 1:15–2:05 — uncertain case

Upload an ambiguous object chosen during the 20-run evaluation. Show the LangGraph pause and top
three options. Pick the correct class. Explain both thresholds and show `confirm_prediction` in the
trace after resume.

## 2:05–2:35 — incorrect/correction case

Use the correction control on a completed result. Show that the selected class replaces the model
prediction and causes a fresh city-filtered retrieval. Explain why human correction is part of the
product rather than hidden as a model failure.

## 2:35–3:00 — engineering close

Show the measured test report and confusion matrix. Mention the local Qwen fallback, no image
storage, offline tests, Colab training and Space deployment. End with one limitation and one next
step: locally reviewed translations or expanded special-waste classes.

