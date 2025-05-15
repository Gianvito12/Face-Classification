import sys
from data_cleaner import scale_crop_discard
from skimage.feature import local_binary_pattern
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) != 3:
        print("Errore passaggio parametri")
        return None

    if sys.argv[2] == "0":
        img_result = scale_crop_discard(sys.argv[1])
    elif sys.argv[2] == "1":
        img_result = cv2.imread(sys.argv[1])
    else:
        return 
    
    with open("best_model.pkl", "rb") as f:
        best_model = pickle.load(f)

    img_result_gray = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
   
    if "uniform" in best_model["test_path"]:
        bins=10
        lbp = local_binary_pattern(img_result_gray, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp, bins=bins, density=True)
    else:
        bins=256
        lbp = local_binary_pattern(img_result_gray, P=256, R=1, method="default")
        hist, _ = np.histogram(lbp, bins=bins, density=True)
    
    if best_model['scaled']:
        hist_to_predict=best_model['scaler'].transform([hist])
    else:
        hist_to_predict=hist

    y_pred = best_model['model'].predict(hist_to_predict)

    
    print(f"Predicted Class: {y_pred[0]}")

    
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    
    axes[0].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Predicted class: { 'Fake' if int(y_pred[0]) == 1 else 'Real'}")
    axes[0].axis("off")

    axes[1].imshow(np.uint8(lbp),cmap="gray")
    axes[1].set_title("LBP image")
    axes[1].axis('off')

    axes[2].bar(np.arange(len(hist_to_predict[0])),hist_to_predict[0],width=0.8,color="green",edgecolor="black")
    axes[2].set_title("LBP Histogram")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
