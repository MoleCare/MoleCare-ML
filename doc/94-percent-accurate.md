# Our melanoma classifier is 94% accurate. That number is close to meaningless.

We published the model, the weights, the training notebooks and the model card
for MoleCare's melanoma classifier this month. The headline number is 94.2%
test accuracy. This post is about why we put a section in the model card
explaining that you should not trust it, and what we are asking for instead.

## What the model is

An Xception backbone, ImageNet-pretrained and fine-tuned on dermoscopic images
from the [ISIC Archive](https://www.isic-archive.com/). Binary classification —
melanoma vs not-melanoma — on 299x299 RGB input, about 20.9M parameters, served
from Flask. No MoleCare user photos were used in training and none are in the
repository.

We trained five architectures for 50 epochs at batch size 16 and compared them:

| Model | Test accuracy | Test loss | Params (M) |
|---|---|---|---|
| **Xception** *(deployed)* | **0.9422** | 0.1777 | 20.9 |
| InceptionV3 | 0.9416 | 0.1848 | 21.9 |
| InceptionResNetV2 | 0.9384 | 0.2136 | 54.4 |
| DenseNet201 | 0.9369 | 0.1974 | 18.4 |
| VGG16 | 0.8270 | 0.4212 | 14.7 |

Xception won by six ten-thousandths over InceptionV3. That gap is noise. We
deployed it anyway, which is a defensible engineering call and an indefensible
scientific one, and it is worth being clear about which is which.

## The problem with 94%

We measured accuracy. We did not measure sensitivity, specificity, or AUC-ROC.

For a melanoma classifier, that is the wrong metric to have measured. Accuracy
is the fraction of predictions that were correct, pooled across both classes.
On a class-imbalanced dataset — and lesion datasets are imbalanced — a model can
post a strong accuracy figure while missing a large share of the actual
melanomas, because the negative class dominates the average.

The two errors are not symmetric. A false positive sends someone to a
dermatologist who tells them the mole is fine. A false negative tells someone
with a melanoma that they are fine. Accuracy weighs those identically. The
metric that captures the error we actually care about is sensitivity, and we do
not have it.

So the honest reading of "94.2% accurate" is: we do not know how often this
model misses melanomas. It could be excellent. We have not checked.

The model card now records the thresholds we think should gate any future
deployment:

| Metric | Why | Proposed minimum |
|---|---|---|
| Sensitivity (recall) | Missing a melanoma is the harmful failure | >= 0.85 |
| AUC-ROC | Best single discrimination measure | >= 0.90 |
| Specificity | Limits unnecessary biopsies and alarm | >= 0.80 |

Those are targets, not results. Nothing in the repository has cleared them yet.

## The bigger gap: skin tone

Public dermoscopic datasets — ISIC, HAM10000, and the derivatives most published
melanoma models are trained on — are drawn overwhelmingly from lighter-skinned
populations, largely because they were assembled from clinics in Europe,
Australia and North America. Models trained on them inherit the skew.

Our model's performance across Fitzpatrick skin types is **unmeasured**. Not
"acceptable", not "slightly worse" — unmeasured. We have not broken the test set
down by skin type, and the source data may not carry the labels needed to do it
cleanly.

This is not a neutral gap in a spreadsheet. Melanoma on darker skin is already
diagnosed later and carries worse outcomes. It presents differently — more often
acral, on palms, soles and nail beds, in places dermoscopic archives
under-sample twice over. A classifier that quietly performs worse on Fitzpatrick
V-VI does not merely fail to help. It takes an existing inequity and gives it a
confident numerical output.

There is a version of this post that does not exist, where we ran the stratified
evaluation first and reported the numbers. We would rather publish the gap than
sit on the model until we have closed it, because the gap is the interesting
part and someone else may close it faster than we will.

## What else is wrong with it

Since the point is to be useful rather than flattering:

- **No calibration.** The output score is not a calibrated probability. It
  should never be shown to a person as "72% likely melanoma", and MoleCare's
  apps do not show it at all.
- **Domain mismatch.** It was trained on dermoscopic images — contact
  dermatoscope, controlled lighting, standardised scale. Consumer phone photos
  differ on every one of those axes. Performance on phone photos is not
  characterised.
- **Binary framing.** Melanoma vs not-melanoma collapses basal cell carcinoma,
  squamous cell carcinoma, actinic keratosis and ordinary nevi into one negative
  class. The model cannot tell you which of those it thinks it is looking at.
- **An output-polarity footgun.** The raw sigmoid output is P(NotMelanoma), not
  P(Melanoma). If you load the weights and read the score directly you will get
  a confidently inverted answer. Use the `melanoma_probability()` helper. This
  is documented now because we would rather you know than discover it.
- **Keras version lock.** The released artefact is a Keras 2.9 SavedModel. Keras
  3 cannot read it; you need `tf-keras` installed.

## Why publish this

Three reasons, none of them altruistic.

The first is that a model card that only lists strengths is marketing. The
[Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) framework
exists because the failure modes are the part practitioners need, and in medical
imaging specifically, the published-accuracy-to-real-performance gap has a long
history. We would rather be legible than impressive.

The second is that MoleCare is not a diagnostic product and this model does not
diagnose anything in it. The apps are a photo journal — track a mole over time,
compare images, export a PDF for a clinician. There is no triage logic, no
urgency score, no threshold that tells someone to seek care. That boundary is
written into the contributing guide as a hard rule: a pull request that improves
a benchmark number but removes a disclaimer gets declined. Having no clinical
claim to defend makes it much cheaper to be honest about the model.

The third is that we want the bias evaluation done, and we cannot credibly ask
for it while presenting the model as finished.

## The ask

The [open issue](https://github.com/MoleCare/MoleCare-ML/issues/10) is
`bias-evaluation`, and the useful contributions are:

- **Stratified evaluation** — accuracy, sensitivity, specificity and AUC broken
  down by Fitzpatrick type instead of pooled.
- **Dataset composition analysis** — what is actually in the training set by
  skin type, and what is missing. Establishing the baseline honestly is
  valuable on its own.
- **Per-group calibration** — a confidence score that means different things for
  different skin types is worse than no score.
- **Mitigation experiments** — reweighting, targeted augmentation, curriculum
  choices, or sourcing better data.

The ground rule, stated in the issue: work that shows the model performs badly
somewhere is as valuable as work that improves it. Unflattering results are the
ones we most want in the open, and publishing the notebooks is pointless if the
answer is only allowed to be good news.

Weights are in Releases. Code and weights are Apache-2.0; the ISIC training data
carries its own per-collection licences, some non-commercial, so check before
redistributing anything derived from it.

**MoleCare is not a medical device and this model is not a diagnostic tool.** It
has no clinical validation and no regulatory approval. If you are worried about
a mole, see a clinician.

- Model: [github.com/MoleCare/MoleCare-ML](https://github.com/MoleCare/MoleCare-ML)
- Model card: [MODEL_CARD.md](https://github.com/MoleCare/MoleCare-ML/blob/main/MODEL_CARD.md)
- Bias issue: [#10](https://github.com/MoleCare/MoleCare-ML/issues/10)
