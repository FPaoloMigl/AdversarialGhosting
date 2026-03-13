# _**QUALITY ASSESSMENT CODE**_

## **STEPS:**

**1.** Run the full pipeline with the initial dataset  
**2.** Run the full pipeline with the initial dataset but top-k (50 or 1000)  
**3.** Run the full pipeline with a new dataset with same size wall as 1. but with orientation  
**4.** Run 3. but with also top-k  
**5.** Run full pipeline with a smaller wall and random orientation as previous steps  
**6.** Run 5. with also top-k (50 or 1000)  
**7.** Comapre this with the results obtained before this

---

## _**Run the full pipeline with the initial dataset**_

1.  Run code `create_dataset_v2.py`
    1.  Size wall: `scale: [0.1, 6.0, 5.0]`
    2.  `position: [z/2]` to stay on the ground
    3.  `output: .ply file and pkl file`
2.  Convert .ply file into npy using `ply2npy.py inside utils folder`
3.  Retrieve the feature for each layer using `demo_with_output_layer_v3.py`
    1.  `output: .pth files`
4.  Calculate the metrics to choose the best layer to create the training dataset for the model using the script `compute_metrics`
    1.  `output: cvs file with all the metrics for each layers`
5.  Analyze layers also using `plot_layerwise_curves.py` (_for this check if we have to use max or mean)_ and `line_chart.py`
6.  Choose the layer that's most affected by the perturbation
    1.  Create the training dataset using the script `prepare_surrogate_data.py`
        1.  Compute the feature shock using difference
        2.  `output: surrogate_training_data_... .pt` file
    2.  Choose the surrogate architecture
    3.  Train the surrogate model using `train_surrogate_model.py`
        1.  `output: .png file, pth file`
    4.  Run optimization
        1.  \`output: theta\_convergence\_plot\_v2.png file \`