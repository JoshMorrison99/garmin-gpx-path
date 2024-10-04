import argparse
import numpy as np
from datetime import datetime
import xmltodict
from PIL import Image, ImageDraw
import os
from concurrent.futures import ThreadPoolExecutor

# Convert timestamp to seconds relative to start_time
def get_seconds(timestamp, start_time):
    fmt = '%Y-%m-%dT%H:%M:%S.%fZ'  # Adjust based on the actual format in your GPX file
    time_obj = datetime.strptime(timestamp, fmt)
    start_time_obj = datetime.strptime(start_time, fmt)
    return (time_obj - start_time_obj).total_seconds()

def parse_gpx(data):
    gpx = []
    longitudes = []
    latitudes = []

    start_time = data['gpx']['metadata']['time']

    for pulse in data['gpx']['trk']['trkseg']['trkpt']:
        gpx.append({
            'longitude': pulse['@lon'], 
            'latitude': pulse['@lat'], 
            'elevation': pulse['ele'],
            'timestamp': pulse['time'],
            'heart_rate': pulse['extensions']['ns3:TrackPointExtension']['ns3:hr'],
            'cadence': pulse['extensions']['ns3:TrackPointExtension']['ns3:cad']
        })
        longitudes.append(float(pulse['@lon']))
        latitudes.append(float(pulse['@lat']))

    return gpx, longitudes, latitudes, start_time

# Function to create and save a single frame
def create_frame(t, width, height, coords, times, output_path):
    # Create a transparent background
    frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(frame)

    # Draw the path
    for i in range(len(coords) - 1):
        start_point = (int(coords[i][0]), int(coords[i][1]))
        end_point = (int(coords[i + 1][0]), int(coords[i + 1][1]))
        draw.line([start_point, end_point], fill=(255, 255, 255, 255), width=6)  # White line for the path

    # Find the closest index to the current time t
    closest_index = np.argmin(np.abs(times - t))
    x, y = coords[closest_index]

    # Draw the moving dot (red)
    x = int(x)
    y = int(y)
    draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(255, 0, 0, 255))  # Red dot

    # Save the frame as a PNG file
    frame.save(os.path.join(output_path, f"frame_{int(t * 1000):06d}.png"))

def main():
    parser = argparse.ArgumentParser(description="Generate PNG frames based on GPX data.")
    parser.add_argument("-f", "--file", type=argparse.FileType('r'), help="Path to the file to be processed", required=True)
    parser.add_argument("-o", "--output", type=str, default="frames", help="Output directory for PNG frames")
    args = parser.parse_args()

    xml_content = args.file.read()
    data = xmltodict.parse(xml_content)

    gpx, longitudes, latitudes, start_time = parse_gpx(data)

    min_lat = min(latitudes)
    max_lat = max(latitudes)
    min_lon = min(longitudes)
    max_lon = max(longitudes)

    # Calculate the range of latitudes and longitudes
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    # Set one dimension, and calculate the other dynamically based on the range
    base_height = 1080  # Set the base height
    aspect_ratio = lon_range / lat_range  # Calculate the aspect ratio

    # Calculate the corresponding width to preserve aspect ratio
    width = int(base_height * aspect_ratio)
    height = base_height

    # Map lat/lon to pixel coordinates
    def map_to_pixel(lat, lon):
        x = (lon - min_lon) / (max_lon - min_lon) * width
        y = (lat - min_lat) / (max_lat - min_lat) * height
        return x, y

    # Convert lat/lon to pixel coordinates
    coords = np.array([map_to_pixel(float(obj['latitude']), float(obj['longitude'])) for obj in gpx])
    times = np.array([get_seconds(obj['timestamp'], start_time) for obj in gpx])

    # Get the total duration of the video
    duration = max(times) - min(times)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Create PNG frames using multithreading
    with ThreadPoolExecutor() as executor:
        time_range = np.arange(min(times), max(times), 1 )  # 1 FPS
        futures = [executor.submit(create_frame, t, width, height, coords, times, args.output) for t in time_range]

        # Wait for all threads to complete
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
