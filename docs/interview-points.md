# Interview points

- **Why does a too-large learning rate fail?** Each step overshoots the minimum by more than it
  corrects; here step 0.12 makes the loss grow every step.
- **How do you spot overfitting?** Training error keeps falling while held-out error rises; degree 12
  had the lowest training RMSE and by far the worst held-out RMSE.
- **Why do hidden layers matter?** A linear model cannot combine strokes; one ReLU layer lifted shape
  accuracy from about 82% to about 99%.
- **What does softmax do to logits?** Exponentiate and normalise, after subtracting the max for
  numerical safety.
- **Why divide attention scores by sqrt(d)?** Dot products grow with dimension; without scaling,
  softmax saturates and gradients vanish.
- **What does temperature change?** Only the sharpness of the next-token distribution; at T = 0 the
  20 samples were identical, at 1.5 there were 17 different ones.
- **What does SFT add over pretraining?** The answer format. Facts mostly come from pretraining, and
  here they only partly carried over (2/6).
- **How does a diffusion model generate?** It learns to predict added noise, then removes predicted
  noise step by step starting from pure noise.
- **How do you stop an agent from making things up?** Require each claim to cite a retrieved passage
  or a real tool result, and drop the rest; refuse when the budget cannot be met.
- **What is MCP in one line?** A shared JSON-RPC contract (`tools/list`, `tools/call`) so any agent can
  use any tool server without custom glue.
