import cv2
import pandas as pd
import numpy as np

def apply_tracking(img, frame, pos, scale, rot, opacity):
    # Calculate the transformation matrix
    M = cv2.getRotationMatrix2D(tuple(pos), rot, min(scale))
    M[0, 2] += frame.shape[1] / 2 - img.shape[1] / 2
    M[1, 2] += frame.shape[0] / 2 - img.shape[0] / 2

    # Warp the image to match the tracking data
    warped_img = cv2.warpAffine(img, M, (frame.shape[1], frame.shape[0]))

    # Add opacity
    mask = cv2.warpAffine(np.ones(img.shape[:2], dtype=np.uint8), M, (frame.shape[1], frame.shape[0]))
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert the mask to BGR format
    mask = np.where(mask > 0, opacity, 0)  # Apply opacity

    # Overlay the image on the frame
    frame = cv2.addWeighted(frame, 1, warped_img, mask / 255.0, 0)

    return frame

def main(image_path, video_path, tracking_data_path, output_path):
    # Load the tracking data
    tracking_data = pd.read_csv(tracking_data_path, header=None, names=["pos_x", "pos_y", "scale_x", "scale_y", "rotation", "opacity"])

    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Open the video
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Process each frame of the video
    for i, row in tracking_data.iterrows():
        ret, frame = video.read()
        if not ret:
            break

        frame = apply_tracking(img, frame, (row["pos_x"], row["pos_y"]), (row["scale_x"] / 100.0, row["scale_y"] / 100.0), row["rotation"], row["opacity"] / 100.0)
        writer.write(frame)

    video.release()
    writer.release()

if __name__ == "__main__":
    # Adjust the paths as needed
    image_path = "path_to_your_image.png"
    video_path = "path_to_your_video.mp4"
    tracking_data_path = "path_to_your_tracking_data.csv"
    output_path = "output_video.mp4"
    main(image_path, video_path, tracking_data_path, output_path)
