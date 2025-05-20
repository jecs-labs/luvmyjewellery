################################   Libraries   ################################

import os.path
import uuid

import cv2
import loader

#############################   Global variables   ############################

# Camera source configuration
VID_SOURCE = "USB_CAM"  # Only USB_CAM is supported in browser environment

# Change to True if you need to flip image
NEED_FLIP = False

# Default width & height of the images
WIDTH = 720
HEIGHT = 640

# Store the current jewellery information
jewellery = None
impose = None
mx = my = None
dw = dh = None


#############################   Helper functions   ############################

def update_impose():
    """
    Update the new jewellery info when previous or next button is pressed.

    Returns:
        None
    """
    global impose, mx, my, dw, dh
    impose = jewellery["path"]
    mx, my = jewellery["x"], jewellery["y"]
    dw, dh = jewellery["dw"], jewellery["dh"]


def save_picture(curr_frame):
    """
    Save the current frame as JPEG image.

    Args:
        curr_frame (np.array): Current frame when user wanted to save it
    Returns:
        None
    """
    # Create the output folder if not exists
    captures_dir = os.path.join('captures')
    if not os.path.exists(captures_dir):
        os.makedirs(captures_dir)

    filename = os.path.join(captures_dir, f'Capture_{str(uuid.uuid4())[:6]}.jpg')
    # Get a unique filename
    while os.path.exists(filename):
        filename = os.path.join(captures_dir, f'Capture_{str(uuid.uuid4())[:6]}.jpg')
    cv2.imwrite(filename, curr_frame)


###############################   Main Program   ##############################

if __name__ == "__main__":
    # Get the source from settings configuration file
    cam = cv2.VideoCapture(loader.get_source(VID_SOURCE))
    cv2.imshow("preview", loader.generate_loading_screen(HEIGHT, WIDTH))

    # Load the required files into memory
    cascade, jewelleries = loader.load_files()
    jewel_key_list = list(jewelleries)

    # Set the first jewellery as current jewellery in use
    curr_jewel_index = 0
    jewellery = jewelleries[jewel_key_list[curr_jewel_index]]
    impose = jewellery["path"]
    mx, my = jewellery["x"], jewellery["y"]
    dw, dh = jewellery["dw"], jewellery["dh"]

    # Start reading the frames from source
    # alpha = 0.9 # 0.0 to 1.0
    check, frame = cam.read()
    cv2.waitKey(2000)

    if check:
        # If able to read the source; start capturing frames
        while check:
            check, frame = cam.read()
            if not check:
                continue

            # If you need to flip the image horizontally;
            # just change the NEED_FLIP to True above in global variable section
            if NEED_FLIP:
                frame = cv2.flip(frame, 1)

            # Resize the frame
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            # cv2.imshow('preview', frame)

            # Detect available faces in frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # you can change the scaleFactor & minNeighbors if required
            faces = cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=3)
            # for x, y, w, h in faces:
            #     frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Show the filter, if only 1 face is found otherwise it may create inconsistency
            # You can try with multiple faces... here
            if len(faces) == 1:
                # Extract face location
                x, y, w, h = faces[0]

                # Adjust the jewelery to the face's placement
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                fw, fh = int(w * dw), int(w * dh)
                new_impose = cv2.resize(impose, (fw, fh))

                # Super impose the jewellery over captured frame
                iw, ih, c = new_impose.shape
                for i in range(0, iw):
                    for j in range(0, ih):
                        if new_impose[i, j][3] != 0:
                            if y + i + h + my < HEIGHT and x + j + mx < WIDTH:
                                # print(y + i + h + my, x + j + mx)
                                frame[y + i + h + my, x + j + mx] = new_impose[i, j]

            # Display the final image
            cv2.imshow("preview", frame)

            # Set ESC key to quit the program
            k = cv2.waitKey(15)
            if k == 27:
                break

            # Goto next jewellery when N is pressed
            elif k == ord("n") or k == ord("N"):
                if curr_jewel_index + 1 != len(jewel_key_list):
                    curr_jewel_index += 1
                else:
                    curr_jewel_index = 0
                jewellery = jewelleries[jewel_key_list[curr_jewel_index]]
                update_impose()

            # Goto previous jewellery when P is pressed
            elif k == ord("p") or k == ord("P"):
                if curr_jewel_index - 1 < 0:
                    curr_jewel_index = len(jewel_key_list) - 1
                else:
                    curr_jewel_index -= 1
                jewellery = jewelleries[jewel_key_list[curr_jewel_index]]
                update_impose()

            # Save the picture when S is pressed
            elif k == ord("s") or k == ord("S"):
                save_picture(frame)

    else:
        print("Error: Can't read camera")

    cv2.destroyAllWindows()
    cam.release()