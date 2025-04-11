import cv2
import numpy as np
import pandas as pd
import tifffile


class MultiCellTracker:
    """
    A class to track multiple cells in a video or image sequence. It supports both single-cell
    and multi-cell tracking, including stopping tracking when cells come into contact with each other.
    """

    def __init__(self, scale_factor=0.5, debug=True, output_video="cell_tracking.avi"):
        """
        Initializes the tracker, preparing storage for cells, trackers, and tracking results.

        Parameters:
        - scale_factor: The factor by which to scale down the image for faster processing.
        - debug: A flag to enable or disable debugging features (e.g., displaying intermediate steps).
        - output_video: The name of the video file to save the results.
        """
        self.trackers = []  # List to store trackers for each cell
        self.cells = []  # List to store selected ROI (Region of Interest) for cells
        self.tracking_results = []  # List to store tracking results (cell positions)
        self.scale_factor = scale_factor
        self.debug = debug
        self.output_video = output_video
        self.video_writer = None  # Video writer for saving the output video
        self.trajectories = {}  # Dictionary to store cell movement trajectories
        self.sin_cell_counter = 1  # Counter for single-cell tracking IDs
        self.multi_cell_counter = 1  # Counter for multi-cell tracking IDs

    def preprocess_frame(self, frame):
        """
        Preprocesses a frame by normalizing the pixel values and resizing it.

        Parameters:
        - frame: The current frame to be processed.

        Returns:
        - The processed frame ready for tracking.
        """
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR if needed
        frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)  # Resize frame
        return frame

    def select_cells(self, first_frame):
        """
        Allows the user to manually select regions of interest (ROI) for tracking cells in the first frame.
        It supports both single-cell and multi-cell selection.

        Parameters:
        - first_frame: The first frame where the cells will be manually selected.
        """
        while True:
            roi = cv2.selectROI(
                "choose cells (Press ESC to exit, Enter to confirm single cell, M to confirm multi-cell)",
                first_frame, fromCenter=False, showCrosshair=True)  # Allow ROI selection
            x, y, w, h = roi
            if w < 10 or h < 10:
                print("❌ The selected ROI is too small. Please select again.")
                continue

            key = cv2.waitKey(0)  # Wait for user input

            if key == 27:  # ESC key to exit
                break
            elif key == 13:  # Enter key to confirm single-cell
                print("✅ Single Cell Confirmed:", roi)
                self.cells.append((roi, f"SinCell {self.sin_cell_counter}"))
                self.sin_cell_counter += 1
                tracker = cv2.legacy.TrackerCSRT_create()  # Create a CSRT tracker for the selected cell
                tracker.init(first_frame, roi)  # Initialize the tracker with the first frame and selected ROI
                self.trackers.append(tracker)
                self.trajectories[len(self.trackers)] = []  # Initialize the trajectory for this tracker
            elif key == ord('m'):  # M key to confirm multi-cell
                print("✅ Multi Cell Confirmed:", roi)
                multi_cell_id = f"MultiCell {self.multi_cell_counter}"
                self.multi_cell_counter += 1
                self.cells.append((roi, multi_cell_id))
                tracker = cv2.legacy.TrackerCSRT_create()  # Create a CSRT tracker for the selected multi-cell
                tracker.init(first_frame, roi)  # Initialize the tracker for multi-cell
                self.trackers.append(tracker)
                self.trajectories[len(self.trackers)] = []  # Initialize the trajectory for this tracker

        cv2.destroyAllWindows()  # Close the ROI selection window after finishing

    def is_contact(self, bbox1, bbox2, threshold=20):
        """
        Determines if two cells are in contact by calculating the overlap between their bounding boxes.

        Parameters:
        - bbox1: The bounding box of the first cell.
        - bbox2: The bounding box of the second cell.
        - threshold: The threshold distance for considering cells to be in contact.

        Returns:
        - True if the cells are in contact, otherwise False.
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate overlap between two bounding boxes
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

        if x_overlap > 0 and y_overlap > 0:  # If there is overlap, consider them as in contact
            return True
        return False

    def track_cells(self, ome_path, output_excel):
        """
        Tracks the selected cells across all frames in a given image sequence (OME-TIFF file).

        Parameters:
        - ome_path: The path to the OME-TIFF file containing the image sequence.
        - output_excel: The path to save the tracking results in an Excel file.
        """
        print("Reading OME-TIFF File...")
        images = tifffile.imread(ome_path)  # Load the OME-TIFF images

        if images.ndim == 2:  # If the image is 2D, add an extra dimension
            images = images[np.newaxis, ...]

        print(f"Image Dimensions: {images.shape}")
        first_frame = self.preprocess_frame(images[0])  # Preprocess the first frame
        self.select_cells(first_frame)  # Let the user select cells in the first frame

        frame_height, frame_width = first_frame.shape[:2]
        self.video_writer = cv2.VideoWriter(self.output_video, cv2.VideoWriter_fourcc(*'XVID'), 10,
                                            (frame_width, frame_height))  # Set up the video writer

        frame_index = 0
        active_trackers = list(range(len(self.trackers)))  # Track active trackers
        to_remove = []  # List to store cells that need to be removed due to contact
        for frame in images:
            frame = self.preprocess_frame(frame)  # Preprocess the frame

            # Update all trackers with the new frame
            bboxes = []
            for tracker in self.trackers:
                success, bbox = tracker.update(frame)
                if success:
                    bboxes.append(bbox)
                else:
                    bboxes.append(None)  # If update fails, mark as None

            # Check for contact between cells, and remove them if they are in contact
            for i, bbox1 in enumerate(bboxes):
                if bbox1 is None:
                    continue

                roi, cell_id = self.cells[i]
                for j, bbox2 in enumerate(bboxes):
                    if i != j and bbox2 is not None:  # Don't compare the same cell
                        other_roi, other_cell_id = self.cells[j]
                        if self.is_contact(bbox1, bbox2):  # If the cells are in contact
                            print(f"✅ {cell_id} and {other_cell_id} are in contact, stop tracking")
                            to_remove.extend([i, j])

            # Only record the trajectory data for SinCells (single-cell)
            for i, bbox in enumerate(bboxes):
                if bbox is None or i in to_remove:
                    continue

                roi, cell_id = self.cells[i]
                x, y, w, h = [int(v) for v in bbox]
                cx, cy = x + w // 2, y + h // 2  # Get the center coordinates of the bounding box

                if "SinCell" in cell_id:
                    self.tracking_results.append([cell_id, frame_index, cx / self.scale_factor, cy / self.scale_factor])

                    # Save the trajectory data
                    self.trajectories[i + 1].append((cx, cy))

                    # Draw the tracking path
                    for j in range(1, len(self.trajectories[i + 1])):
                        cv2.line(frame, self.trajectories[i + 1][j - 1], self.trajectories[i + 1][j], (255, 0, 0), 2)

                    # Draw bounding box and center point
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, cell_id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw bounding box for multi-cells
                if "MultiCell" in cell_id:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame, cell_id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Remove the contact cells from active trackers
            active_trackers = [i for i in active_trackers if i not in set(to_remove)]

            frame_index += 1
            self.video_writer.write(frame)  # Write the frame to the output video
            cv2.imshow("Tracking", frame)  # Display the frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.video_writer.release()

        # Save the tracking results (only SinCells) to an Excel file
        df = pd.DataFrame(self.tracking_results, columns=["ID", "Frame", "X", "Y"])
        df = df.sort_values(by=["ID", "Frame"])

        # Remove SinCells that came into contact
        df = df[~df["ID"].isin([f"SinCell {i + 1}" for i in set(to_remove)])]

        df.to_excel(output_excel, index=False)
        print(f"✅ Tracking completed. Cell trajectories saved to {output_excel}")
        print(f"✅ Video saved to {self.output_video}")


if __name__ == "__main__":
    # Initialize the tracker and specify the paths for the input and output files
    tracker = MultiCellTracker(scale_factor=0.5, debug=True,
                               output_video="C:\\Users\\ykq51\\Desktop\\PythonProject\\cell_tracking.avi")
    ome_file_path = r"C:\Users\ykq51\Desktop\PythonProject\NPC43_XY07.ome.tif"
    output_excel_path = r"C:\Users\ykq51\Desktop\PythonProject\sin_cell_tracking.xlsx"
    tracker.track_cells(ome_file_path, output_excel_path)

