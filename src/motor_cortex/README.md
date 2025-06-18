![](../../imgs/stimulated-coordinated-movement.png)

This is a body of work that makes predictions of movement of a rhesus macaque monkey on the bases of raw ECoG data from the primary motor cortex.

The monkey's right wrist is predicted as ipsilateral movement because the monkey was reaching for food on the right side of its body with the right wrist.

This monkey's left wrist is predicted as contralateral movement because the monkey was reaching for food on the right side of its body with its left wrist.

The movement truth, predictions, and visualized results as will as the model validation performance is provided below

## Results:

**Validation performance of the model: Hybrid_CNN_LSTM_contralateral_3_output_session_0:**

```
Contralateral:
Epoch 95/100 | Train Loss: 0.002427 | Val Loss: 0.004928 | R2: 0.926363
```

```
Ipsiplateral
R2: 0.60+
```

![](/imgs/contralateral_validation_performance.png)

Contralateral Motion Truth:

![](/imgs/Contralateral_Motion.png)

Contralateral Motion Prediction:

![](/imgs/Contralateral_Motion_Prediction.png)

---

Ipsilateral Motion Truth:

![](/imgs/Session1Movement.png)

Ipsilateral Motion Prediction:

![](/imgs/Session1PredictedMovement.png)
