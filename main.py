import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'model1.pth'
pretrained_vit = torch.load(model_path, map_location=device, weights_only=False)
pretrained_vit.eval().to(device)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
class_labels = {
    0: "Neutral",
    1: "Basal Rot Detected"
}

cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()  
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    img_tensor = image_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        model_output = pretrained_vit(img_tensor)

    probabilities = torch.nn.functional.softmax(model_output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

    prediction_label = class_labels.get(predicted_class, "Unknown")

    cv2.putText(frame, f"Predicted: {prediction_label}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

   

    cv2.imshow("Live Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
