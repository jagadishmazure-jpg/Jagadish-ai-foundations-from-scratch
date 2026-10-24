# How it works

## The ladder

Artificial intelligence is the broad goal: software doing jobs we used to think needed a person.
Machine learning is the part of that where the behaviour is fitted from examples instead of typed
in. Deep learning is machine learning with many stacked layers of simple units. Generative models
are deep networks trained to produce new samples (text, pictures) rather than labels. Each rung
below is a module.

## 1. Rules vs learned (`classic.rules_vs_learned`)

A hand-written filter flags three trigger words. A logistic model is fitted on 1,500 labelled
messages. The fitted weights rediscover the trigger words and also pick up the ones the rule
forgot, which is why it wins on the 500 held-out messages.

## 2. The training loop and gradient descent (`classic.gradient_descent`)

Predict, measure mean squared error, compute which way each weight should move to shrink the
error, move a small step, repeat. The sweep shows the step size trade-off: too small crawls,
too big overshoots and the loss explodes.

## 3. Fitting just right (`classic.fit_sweep`)

Fifteen noisy points from a gentle curve. A constant (degree 0) cannot follow it; degree 2 matches
the true shape; degree 12 threads every training point and is wildly wrong between them. Training
error keeps falling while held-out error turns back up: that gap is overfitting.

## 4. Learning without labels (`classic.kmeans`)

Laundry items described only by length and weight. k-means never sees the item type, yet its three
groups line up with shirts, trousers and socks.

## 5. Data volume (`classic.learning_curve`)

Same spam model, 10 to 5,000 training examples. Accuracy climbs steeply then flattens: more data
helps a lot until the model's capacity, not the data, is the limit.

## 6. Neural networks (`neural`)

A unit multiplies inputs by weights, adds them, and passes the result only if positive (ReLU).
Layers of units feed the next layer. The last layer outputs raw scores (logits); softmax turns
them into probabilities. Backprop pushes the error back through each layer with the chain rule.
On 8x8 bar/box/X pictures, a linear model reaches about 82% and one hidden layer reaches about 99%,
because a box vs an X needs combinations of strokes that a single layer cannot express.

## 7. Embeddings (`embeddings`)

Count which words appear near which in a generated corpus, reweight the counts so common words do
not dominate (positive PMI), and compress with SVD. Words used in similar company end up with
similar vectors; a relation applied consistently (male to female) becomes a consistent offset.

## 8. Attention and transformers (`attention`, `lm`)

Each position asks a question (query), advertises itself (key) and carries content (value). Scores
are query-key dot products scaled by sqrt(d); a mask blocks future positions; softmax gives weights;
the output is the weighted mix of values. A block is attention plus a small feed-forward network,
each with a residual connection. The tiny model stacks 1 or 2 blocks over characters.

## 9. Generation and temperature (`lm.generate`)

Run the model, take the distribution over the next character, pick one, append, repeat. Dividing
logits by the temperature before softmax sharpens (T < 1) or flattens (T > 1) the distribution;
T = 0 means always take the top choice.

## 10. Pretraining then fine-tuning (`lm.run`)

Stage 1 trains on plain sentences about eight places. Stage 2 trains briefly on question/answer
lines for five of them. We then ask about the other three. After stage 1 the model just continues
prose; after stage 2 it answers in the short Q/A shape, and for some held-out places it carries the
stage-1 fact into that shape.

## 11. Diffusion (`diffusion`)

Training: take a clean point on a ring, mix in a known amount of noise, train a network to predict
that noise. Sampling: start from pure noise and subtract a little predicted noise at each of 50
steps. The points drift back onto the ring.

## 12. Multimodal (`multimodal`)

Two encoders, one for 8x8 pictures and one for captions, map into the same 32-number space. The
loss pulls each picture toward its own caption and away from the others in the batch. Afterwards
the nearest caption to an unseen picture is usually the right one.

## 13. Agents, tools, retrieval and MCP (`trip`)

See [trip-walkthrough.md](trip-walkthrough.md).
