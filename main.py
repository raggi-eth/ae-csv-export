import cv2
import pandas as pd
import numpy as np

## with blending and warping

# def apply_tracking(img, frame, pos, scale, rotation, opacity):
#     rows, cols = img.shape[:2]

#     # Compute the rotation matrix
#     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
    
#     # Apply the rotation and scaling to the image
#     img = cv2.warpAffine(img, M, (cols, rows))
    
#     # Scale the image
#     img = cv2.resize(img, None, fx=scale[0], fy=scale[1], interpolation=cv2.INTER_CUBIC)

#     # Create a mask from the image
#     mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply the opacity to the image
#     img = cv2.addWeighted(img, opacity, np.zeros_like(img), 1 - opacity, 0)

#     # Get the new size of the image
#     rows, cols = img.shape[:2]

#     # Compute the translation matrix
#     M = np.float32([[1, 0, pos[0] - cols / 2], [0, 1, pos[1] - rows / 2]])
    
#     # Apply the translation to the image and mask
#     img = cv2.warpAffine(img, M, (frame.shape[1], frame.shape[0]))
#     mask = cv2.warpAffine(mask, M, (frame.shape[1], frame.shape[0]))

#     # Convert the image to have 3 channels if it doesn't already have them
#     if len(img.shape) == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
#     # Warp the input image to the frame
#     warped_img = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
#     warped_img = cv2.add(warped_img, img)

#     # Blend the frame and the warped image
#     frame = cv2.addWeighted(frame, 1, warped_img, 1, 0)

#     return frame



## working function - semi 
# def apply_tracking(img, frame, pos, scale, rotation, opacity):
#     # Apply the opacity to the image
#     img = cv2.addWeighted(img, opacity, np.zeros_like(img), 1 - opacity, 0)

#     # Compute the rotation matrix
#     M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), -rotation, 1)

#     # Apply the rotation to the image
#     img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

#     # Scale the image
#     img = cv2.resize(img, None, fx=scale[0]*4, fy=scale[0]*4, interpolation=cv2.INTER_CUBIC)

#     # Compute the translation matrix
#     M = np.float32([[1, 0, pos[0] - img.shape[1] / 2], [0, 1, pos[1] - img.shape[0] / 2]])

#     # Apply the translation to the image
#     img = cv2.warpAffine(img, M, (frame.shape[1], frame.shape[0]))

#     # Create a mask from the image
#     mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Apply the mask to the frame
#     masked_frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

#     # Add the image to the frame
#     frame = cv2.add(masked_frame, img)

#     return frame

def apply_tracking(img, frame, pos, scale, rotation, opacity):
    # Apply the opacity to the image
    img = cv2.addWeighted(img, opacity, np.zeros_like(img), 1 - opacity, 0)

    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, rotation*-1, 1.0)


    # Apply the rotation to the image
    img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    # Scale the image
    img = cv2.resize(img, None, fx=scale[0]*5, fy=scale[0]*5, interpolation=cv2.INTER_CUBIC)

    # Compute the translation matrix
    M = np.float32([[1, 0, pos[0] - img.shape[1] / 2], [0, 1, pos[1] - img.shape[0] / 2]])

    # Apply the translation to the image
    img = cv2.warpAffine(img, M, (frame.shape[1], frame.shape[0]))

    # Create a mask from the image
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

    # Add the image to the frame
    frame = cv2.add(masked_frame, img)

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
    image_path = "inputs/input.png"
    video_path = "inputs/input.mp4"
    tracking_data_path = "inputs/input.csv"
    output_path = "output/output_video.mp4"
    main(image_path, video_path, tracking_data_path, output_path)
