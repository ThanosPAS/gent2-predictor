# gent2-predictor


Project title: 

“Cancer prediction and inference optimization on microarray gene expression data
from patients using modern deep neural network architectures"

Description:

Cancer prediction i s an i mportant l ong term goal for the healthcare sector that has
the potential to i mprove patients’ survival while employing existing technologies. Data driven
predictions have not penetrated every day medical practice at l arge, l et alone deep l earning
approaches. Our goal i s to use gene expression data from microarrays as a starting point and
aid i n that direction by i mplementing modern deep l earning models for cancer i nferencing.
We have downloaded and deployed the database GENT2, which contains high quality NCBI
data from both normal persons and patients. We extracted cancer patient data that are
annotated for tumor characteristics and disease staging while categorized by cancer subtype
and containing a meta-analysis on survivability and severity of disease. We have extracted the expression
levels for 22000 genes for 5 of the most deadly sub-types of cancer that we initially assume will
be manageable computationally. One of our goals i s to use as much patient data as we possibly
and realistically can feed to the computing solutions we have at our disposal. For model building
purposes, from each sub-type, we have selected 50 patients while taking care to represent the
full spectrum of each cancer’s characteristics (staging/tissue histology etc). We have also fished
out 50 samples from normal people and the i dea i s to either train a FFN or an attention based
model (for example a transformer) on a multi-class classification problem (Normal and 5 types
of cancer), compare i t against a baseline and optimize the network appropriately.

Milestones:

1. Model architecture assessment: FFNs or attention-based models?
2. Model building,
3. Comparison of the unoptimized model with a baseline and i nitial assessment of our approach,
4. Inference optimization and final comparison.
