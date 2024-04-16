import cv2
import torch
import sign_language
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 28)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 28)


img_counter = 0

model = sign_language.SignLanguageModel()
model.load_state_dict(torch.load("sign_language_model.pth"))

out = {i: chr(i + 65) for i in range(26)}

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    width  = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    box_size = 368
    start = (int((width - box_size) // 2), int((height - box_size) // 2))
    end = (int((width + box_size) // 2), int((height + box_size) // 2))
    box = cv2.rectangle(frame, start, end, (255, 0, 0), 3)
    # cv2.imshow("test", frame)
    cv2.imshow("test", box)
    model.eval()
    with torch.no_grad():
        cropped_frame = frame[start[1]:end[1], start[0]:end[0]]
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(cropped_frame, (28, 28)) / 255.0
        # print(image)
        # plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        # plt.imshow(cropped_frame, cmap='gray', vmin=0, vmax=255)
        # plt.imshow(cv2.cvtColor(frame[start[1]:end[1], start[0]:end[0]], cv2.COLOR_RGB2BGR))
        # plt.show()
        # input()
        # break
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        print(out[predicted.item()])
        cv2.putText(frame, out[predicted.item()], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("test", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
cam.release()
cv2.destroyAllWindows()