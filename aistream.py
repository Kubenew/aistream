import cv2
import tensorflow as tf

# Load the pre-trained MobileNet model
model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Define a function to compress a single frame
def compress_frame(frame):
    # Resize the frame to the desired input shape
    resized_frame = cv2.resize(frame, (224, 224))

    # Convert the frame to a tensor
    frame_tensor = tf.expand_dims(resized_frame, axis=0)

    # Normalize the pixel values to the range [0, 1]
    frame_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(frame_tensor)

    # Compress the frame using the neural network
    compressed_frame_tensor = model(frame_tensor)

    # Convert the compressed frame tensor back to a numpy array
    compressed_frame = compressed_frame_tensor.numpy()[0]

    # Convert the pixel values back to the range [0, 255]
    compressed_frame = ((compressed_frame + 1.0) / 2.0) * 255.0

    # Convert the numpy array back to an OpenCV image
    compressed_frame = compressed_frame.astype('uint8')

    return compressed_frame

# Open the video stream
stream = cv2.VideoCapture('my_video.mp4')

# Loop through the frames in the video stream and compress them in real-time
while True:
    # Read the next frame from the video stream
    ret, frame = stream.read()

    # If the frame could not be read, break out of the loop
    if not ret:
        break

    # Compress the frame using the neural network
    compressed_frame = compress_frame(frame)

    # Display the compressed frame
    cv2.imshow('Compressed Frame', compressed_frame)

    # Wait for a key press and check if the 'q' key was pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
stream.release()
cv2.destroyAllWindows()
