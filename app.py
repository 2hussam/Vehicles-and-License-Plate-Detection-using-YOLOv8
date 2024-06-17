import cv2 as cv
from ultralytics import YOLO
import math
import time
import csv
import threading
import argparse
import os
from pathlib import Path

# Function to check if vehicle data exists in CSV and update if needed
def update_vehicle_data(track_id: int, vehicle_class: str, plate_filename: str, detection_date: str, detection_time: str, vehicle_data_path: str) -> None:
    # Open CSV file for reading
    with open(f"{vehicle_data_path}/vehicle_data.csv", 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        
    # Check if track ID already exists
    existing_data = [row for row in data if row[1] == str(track_id)]

    if existing_data:
        # Update existing row with new data
        existing_index = data.index(existing_data[0])
        data[existing_index] = [vehicle_class, track_id, plate_filename, detection_date, detection_time]
        # Clear CSV file and rewrite updated data
        with open(f"{vehicle_data_path}/vehicle_data.csv", 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(data)
    else:
        # Add new vehicle data
        with open(f"{vehicle_data_path}/vehicle_data.csv", 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([vehicle_class, track_id, plate_filename, detection_date, detection_time])

def process_video(video_path: Path, model_path: Path, np_model_path: Path, vehicle_data_path: Path) -> None:
    """
    Runs main app

    Parameters:
        video_path: video path or camera number
        model_path: Path to the vehicle detection model
        np_model_path: Path to the license plate detection model
        vehicle_data_path: Path to save vehicle data and output video
    """
    start_prog_time = time.time()

    if not (vehicle_data_path / "Plate Images").exists():
        os.mkdir(f"{vehicle_data_path}/Plate Images")

    plates = Path(vehicle_data_path / "Plate Images")

    # Vehicle model path
    model = YOLO(model_path)
    # license Plate model path
    np_model = YOLO(np_model_path)

    # Open CSV file for writing
    with open(f"{vehicle_data_path}/vehicle_data.csv", 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Vehicle Name', 'Track ID', 'License Plate Image', 'Detection Date', 'Detection Time'])
    
    # Read video and initialize variables
    video = cv.VideoCapture(video_path)
    ret, frame = video.read()
    H, W, _ = frame.shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    #out = cv.VideoWriter(f"{vehicle_data_path}/video_out.mp4", fourcc, int(video.get(cv.CAP_PROP_FPS)), (W, H))
    out = cv.VideoWriter(f"{vehicle_data_path}/video_out.mp4", fourcc, 20, (W, H))
    # For scaling the plate image
    scale_factor = 2

    read = True
    vehicles_id = [2, 3, 5, 7]
    vehicles_name = ['0', '1', 'car', 'motorcycle', '4', 'bus', '6', 'truck']

    while read:
        ret, frame = video.read()
        if not ret:
            break

        if ret:
            #start_time = time.time()
            # Process detections and Trucking for the Vehicles
            results = model.track(frame, persist=True)[0]

            for detection in results.boxes.data.tolist():
                x1, y1, x2, y2, track_id, score, class_id = detection

                if int(class_id) in vehicles_id and score > 0.6:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w_vehicle, h_vehicle = x2 - x1, y2 - y1
                    score = math.ceil((score * 100)) / 100

                    # License plate detection region
                    x = 460
                    y = 250
                    xtop = x + 500
                    xleft = x - 260
                    yleft = y + 375
                    xbottom = xleft + 1265

                    """
                    Here we specify the region within which we want to detect the License plates 
                    by drawing lines on the frame that represent a rectangle.
    
                    #top line
                    cv.line(frame, (x, y), (xtop, y), color=(0, 255, 0), thickness = 3)
                    #left tine
                    cv.line(frame, (x, y), (xleft, yleft), color=(0, 255, 0), thickness = 3)
                    #bottom line
                    cv.line(frame, (xleft, yleft), (xbottom, yleft), color=(0, 255, 0), thickness = 3)
                    #right line
                    cv.line(frame, (xbottom, yleft), (xtop, y), color=(0, 255, 0), thickness = 3)
    
                    """
                    # Check if vehicle is within detection region
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    if (xleft < center_x < xtop and y < center_y < yleft):
                        """roi: region of interest"""
                        roi = frame[int(y1):int(y2), int(x1):int(x2)]
                        # Process detections for License Plate
                        license_plates = np_model(roi)[0]
                        for license_plate in license_plates.boxes.data.tolist():
                            plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate
                            if (plate_score > 0.4):
                                plate_x1, plate_y1, plate_x2, plate_y2 = int(plate_x1), int(plate_y1), int(plate_x2), int(plate_y2)
                                w, h = plate_x2 - plate_x1, plate_y2 - plate_y1
                                
                                # Enhance license plate image
                                w *= scale_factor
                                h *= scale_factor
                                plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                                plate = cv.resize(plate, (w, h), interpolation=cv.INTER_CUBIC)
                                timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
                                # Generate unique filename and save image
                                plate_filename = f"{track_id}_plate.jpg"
                                cv.imwrite(plates / plate_filename, plate)
                                # Convert 24-hour time to 12-hour time
                                hour = int(timestamp.split("_")[1].split(":")[0])
                                ampm = "AM" if hour < 12 else "PM"
                                hour = hour % 12 or 12
                                detection_time = f"{hour}:{timestamp.split('_')[1].split(':')[1]}:{timestamp.split('_')[1].split(':')[2]} {ampm}"
                                detection_date = timestamp.split("_")[0]
                                # Update vehicle data in CSV if needed
                                threading.Thread(target=update_vehicle_data, args=(track_id, vehicles_name[int(class_id)], plate_filename, detection_date, detection_time, vehicle_data_path)).start()
                                #update_vehicle_data(track_id, vehicles_name[int(class_id)], plate_filename, detection_date, detection_time, vehicle_data_path)
                                # Merge image with frame
                                frame[y1:y1+h, x1:x1+w] = plate
                    
                    # Calculate text size and position
                    text_size, _ = cv.getTextSize(f'{vehicles_name[int(class_id)]} {int(track_id)}', cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    text_width, text_height = text_size
                    # Adjust rectangle dimensions to include text
                    text_buffer = 10 # Add buffer around text
                    x_text_start = x1
                    y_text_end = y1 - text_buffer

                    # Draw rectangle with same color as vehicle bounding box
                    color = (29, 138, 44) if vehicles_name[int(class_id)] in ['car', 'motorcycle'] else (0, 84, 211)
                    cv.rectangle(frame, (x1, y1, w_vehicle, h_vehicle), color, thickness=2)
                    # Fill rectangle with same color
                    cv.rectangle(frame, (x1 - 2, y_text_end - text_height, text_width, text_height + text_buffer), color, cv.FILLED)
                    # Display vehicle name and ID
                    cv.putText(frame, f'{vehicles_name[int(class_id)]} {int(track_id)}', (x_text_start, y_text_end), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv.LINE_AA)

            #elapsed_time = time.time() - start_time
            #fps = 1 / elapsed_time
            #cv.putText(frame, f"FPS: {fps:.2f}", (10, 155), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Write frame to video and show it
            out.write(frame)
            cv.imshow('frame', frame)
            # Break if 'q' is pressed or the video end
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
    # Release video capture and video writer
    video.release()
    out.release()
    # Close all windows
    cv.destroyAllWindows()
    # Close CSV file
    csv_file.close()

    end_prog_time = time.time() - start_prog_time
    print(f'Total processing time: {end_prog_time:.2f} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vehicle and License Plate Detection')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the vehicle detection model')
    parser.add_argument('--np_model_path', type=str, required=True, help='Path to the license plate detection model')
    parser.add_argument('--vehicle_data_path', type=str, required=True, help='Path to save vehicle data and output video')

    args = parser.parse_args()
    process_video(Path(args.video_path), Path(args.model_path), Path(args.np_model_path), Path(args.vehicle_data_path))


    