import re

log_data = """
Epoch 0 - Mean Loss: 0.2927, Mean IoU: 0.0574
Epoch 1 - Mean Loss: 0.2775, Mean IoU: 0.0810
Epoch 2 - Mean Loss: 0.2761, Mean IoU: 0.1051
Epoch 3 - Mean Loss: 0.2703, Mean IoU: 0.1172
Epoch 4 - Mean Loss: 0.2684, Mean IoU: 0.1172
Epoch 5 - Mean Loss: 0.2660, Mean IoU: 0.1264
Epoch 6 - Mean Loss: 0.2638, Mean IoU: 0.1303
Epoch 7 - Mean Loss: 0.2629, Mean IoU: 0.1317
Epoch 8 - Mean Loss: 0.2605, Mean IoU: 0.1652
Epoch 9 - Mean Loss: 0.2589, Mean IoU: 0.1919
Epoch 10 - Mean Loss: 0.2578, Mean IoU: 0.1837
Epoch 11 - Mean Loss: 0.2562, Mean IoU: 0.1960
Epoch 12 - Mean Loss: 0.2555, Mean IoU:0.2051
Epoch 13 - Mean Loss: 0.2540, Mean IoU: 0.2324
Epoch 14 - Mean Loss: 0.2543, Mean IoU: 0.2132
Epoch 15 - Mean Loss: 0.2519, Mean IoU: 0.2705
Epoch 16 - Mean Loss: 0.2546, Mean IoU: 0.2630
Epoch 17 - Mean Loss: 0.2531, Mean IoU: 0.2626
Epoch 18 - Mean Loss: 0.2516, Mean IoU:0.2939
Epoch 19 - Mean Loss: 0.2508， Mean IoU: 0.3012
Epoch 20 - Mean Loss: 0.2503, Mean IoU: 0.3022
Epoch 21 - Mean Loss: 0.2486, Mean IoU: 0.3008
Epoch 22 - Mean Loss: 0.2505, Mean IoU:  0.3045
Epoch 23 - Mean Loss: 0.2485, Mean IoU: 0.3260
Epoch 24 - Mean Loss: 0.2494, Mean IoU: 0.3103
Epoch 25 - Mean Loss: 0.2477, Mean IoU: 0.3354
Epoch 26 - Mean Loss: 0.2474, Mean IoU: 0.3636
Epoch 27 - Mean Loss: 0.2478, Mean IoU: 0.3402
Epoch 28 - Mean Loss: 0.2469, Mean IoU: 0.2918
Epoch 29 - Mean Loss: 0.2455, Mean IoU: 0.3834
"""

# Extract loss and mean IoU values using regular expressions
loss_values = re.findall(r'Mean Loss: ([\d.]+)', log_data)
miou_values = re.findall(r'Mean IoU: ([\d.]+)', log_data)

# Convert strings to float values
loss_values = [float(val) for val in loss_values]
miou_values = [float(val) for val in miou_values]

# Print or use the extracted values
print("Loss Values:", loss_values)
print("Mean IoU Values:", miou_values)