import cv2
import numpy as np

# Global variables for mouse callback
x_start, y_start = 0, 0
x_end, y_end = 0, 0
drawing = False
bbox_selected = False

def mouse_callback(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, drawing, bbox_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_start, y_start = x, y
        x_end, y_end = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x_end, y_end = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_end, y_end = x, y
        bbox_selected = True

def get_bbox(video_path, txt_path):
    # Load video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return None

    # Get screen size
    screen_width = 1920  # Default screen width
    screen_height = 1080  # Default screen height

    # Calculate scaling factor to fit screen while maintaining aspect ratio
    scale = min(screen_width / frame.shape[1], screen_height / frame.shape[0])
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)

    # Create window and set mouse callback
    window_name = "Select Bounding Box"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)
    cv2.setMouseCallback(window_name, mouse_callback)

    frame_copy = frame.copy()
    
    while True:
        image_to_show = frame_copy.copy()
        
        if drawing:
            # Draw rectangle while selecting
            cv2.rectangle(image_to_show, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        
        if bbox_selected:
            # Calculate bbox in x,y,w,h format
            x = min(x_start, x_end)
            y = min(y_start, y_end)
            w = abs(x_end - x_start)
            h = abs(y_end - y_start)
            
            # Save coordinates to file
            with open(txt_path, "w") as f:
                f.write(f"{x},{y},{w},{h}")
            
            print(f"Saved bounding box coordinates (x,y,w,h): {x},{y},{w},{h}")
            break

        cv2.imshow(window_name, image_to_show)
        
        # Press 'q' to quit without selecting
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--txt_path", type=str, required=True, help="Path to ground truth text file")
    video_path = parser.parse_args().video_path
    txt_path = parser.parse_args().txt_path
    get_bbox(video_path, txt_path)
