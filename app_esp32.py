import requests
import cv2
import torch

import numpy as np
import os
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


# Stream from ESP32
stream_url = "http://192.168.171.49:81/stream"
url='http://iotbegineer.com/api/sensors'


try:
    stream = requests.get(stream_url, stream=True)
    buffer = bytes()

    for chunk in stream.iter_content(chunk_size=8192):
        if not chunk:
            print("Received empty chunk.")
            continue

        buffer += chunk
        start = buffer.find(b'\xff\xd8')
        end = buffer.find(b'\xff\xd9')

        if start != -1 and end != -1:
            jpg_data = buffer[start:end + 2]
            buffer = buffer[end + 2:]

            if len(jpg_data) > 0:
                nparr = np.frombuffer(jpg_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)

                    img_tensor = image_transform(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        model_output = pretrained_vit(img_tensor)

                    probabilities = torch.nn.functional.softmax(model_output[0], dim=0)
                    predicted_class = torch.argmax(probabilities).item()

                    prediction_label = class_labels.get(predicted_class, "Unknown")
                    print(prediction_label)
                    if prediction_label == 'Basal Rot Detected':
                                myobj = {'sensor3': 'Basal rot disease detected'}
                                r = requests.post(url, json=myobj, headers={'username': 'iotbegin265', 'Content-Type': 'application/json'})



                    cv2.putText(frame, f"Predicted: {prediction_label}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                   

                    cv2.imshow("Live Classification", frame)                        

                else:
                    print("Failed to decode frame.")
            else:
                print("Received empty JPEG data.")
        else:
            print("Waiting for complete JPEG data...")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except requests.exceptions.RequestException as e:
    print(f"Error connecting to the stream: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    cv2.destroyAllWindows()
