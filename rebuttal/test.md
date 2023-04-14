$Y$

$CE(Y, Y^*) = CE(Y, f(G^*))$

Dear Reviewer y45i,
Thank you for your valuable comments and feedback. Below are our itemized responses to the comments and questions.


Comment 1: Novelty: Seems like a minor modification to the GIB objective
Reply C1: We respectfully disagree with the reviewer's assessment that our approach is a minor modification to the GIB objective. In fact, our paper reveals the distribution shifting issue in existing GIB methods and addresses the issue involved in the GIB objective.
As described in Section 4.1.1, the original GIB method uses $CE(Y, Y^*) = CE(Y, f(G^*))$, which overlooks the distributional divergence between the original graph $G$ and the explanation subgraph $G^*$ after the processing of the prediction model $f$. We then address this problem by mixing the explanation sub-graphs into a mixed complete graph, which is more similar to $G$ than $G^*$.

Our experiments in Section 5.2 also validate the existence of distribution shifts and the effectiveness of our method.

We believe that our approach is a significant contribution to the field of Bayesian inference. We have addressed a critical problem with the GIB objective in the explainability of Graph Neural Networks and have proposed a new approach that is more robust to distribution shifts. We believe that our approach will be of interest to researchers and practitioners working in the field.
