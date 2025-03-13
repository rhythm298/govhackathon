# Traffic Congestion Prediction and Management System
# Core components with implementation details

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import cv2
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import datetime
import json
from flask import Flask, request, jsonify
import paho.mqtt.client as mqtt
import pymongo
from confluent_kafka import Producer, Consumer, KafkaError

# Configuration parameters
config = {
    'camera_endpoints': [
        'rtsp://traffic_cam1.city.gov:554/stream1',
        'rtsp://traffic_cam2.city.gov:554/stream2',
        # Additional cameras would be listed here
    ],
    'intersections': {
        'intersection_1': {
            'location': {'lat': 23.0225, 'lng': 72.5714},  # Ahmedabad coordinates
            'signals': 4,
            'phases': ['NS', 'EW', 'N_TURN', 'S_TURN', 'E_TURN', 'W_TURN'],
            'adjacent': ['intersection_2', 'intersection_5']
        },
        # Additional intersections would be defined here
    },
    'database': {
        'mongodb_uri': 'mongodb://localhost:27017/traffic_management',
        'timeseries_db': 'influxdb://localhost:8086/traffic_metrics'
    },
    'kafka': {
        'bootstrap.servers': 'localhost:9092',
        'traffic_data_topic': 'raw_traffic_data',
        'prediction_topic': 'traffic_predictions',
        'signal_control_topic': 'signal_control_commands'
    },
    'model_params': {
        'window_size': 24,  # 24 time steps (hours) for prediction
        'prediction_horizon': 6,  # Predict 6 time steps ahead
        'batch_size': 64,
        'epochs': 100
    }
}

# Database connections
client = pymongo.MongoClient(config['database']['mongodb_uri'])
db = client['traffic_management']
traffic_collection = db['traffic_data']
incident_collection = db['incidents']

# Kafka Producer setup for real-time data streaming
kafka_producer_conf = {
    'bootstrap.servers': config['kafka']['bootstrap.servers'],
    'client.id': 'traffic_management_producer'
}
kafka_producer = Producer(kafka_producer_conf)

# 1. Computer Vision Module for Vehicle Detection and Counting
class TrafficVisionSystem:
    def __init__(self, camera_endpoints):
        self.camera_endpoints = camera_endpoints
        self.vehicle_detector = self.load_yolo_model()
        self.vehicle_tracker = cv2.legacy.TrackerKCF_create()
        
    def load_yolo_model(self):
        # Load YOLOv4 or a similar object detection model
        weights_path = 'models/yolov4-tiny.weights'
        config_path = 'models/yolov4-tiny.cfg'
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # Use GPU if available
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        # Get output layer names
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
        return {'net': net, 'output_layers': output_layers}
        
    def process_camera_feed(self, camera_id):
        """Process video feed from a specific camera to detect and count vehicles"""
        cap = cv2.VideoCapture(self.camera_endpoints[camera_id])
        
        vehicle_counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0, 'bicycle': 0}
        congestion_level = 0
        average_speed = 0
        tracked_vehicles = []
        
        ret, frame = cap.read()
        if not ret:
            return None
            
        # Resize for faster processing
        frame = cv2.resize(frame, (416, 416))
        
        # Detect vehicles
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.vehicle_detector['net'].setInput(blob)
        outs = self.vehicle_detector['net'].forward(self.vehicle_detector['output_layers'])
        
        class_ids = []
        confidences = []
        boxes = []
        
        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for vehicle classes (car: 2, truck: 7, bus: 5, etc.)
                vehicle_classes = [2, 5, 7, 3, 1]  # COCO dataset IDs
                if class_id in vehicle_classes and confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-max suppression to remove overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Count vehicles by type
        for i in range(len(boxes)):
            if i in indexes:
                label = str(class_id)
                if class_id == 2:  # Car
                    vehicle_counts['car'] += 1
                elif class_id == 7:  # Truck
                    vehicle_counts['truck'] += 1
                elif class_id == 5:  # Bus
                    vehicle_counts['bus'] += 1
                elif class_id == 3:  # Motorcycle
                    vehicle_counts['motorcycle'] += 1
                elif class_id == 1:  # Bicycle
                    vehicle_counts['bicycle'] += 1
        
        # Calculate congestion level (0-100%)
        # This is a simplified example - actual implementation would consider lane occupancy,
        # vehicle density, and historical baselines
        total_vehicles = sum(vehicle_counts.values())
        max_expected_vehicles = 50  # This would be calibrated for each camera/intersection
        congestion_level = min(100, (total_vehicles / max_expected_vehicles) * 100)
        
        # Estimate average speed using optical flow (simplified)
        # Real implementation would use more sophisticated tracking and speed estimation
        if len(tracked_vehicles) > 0:
            average_speed = 30 - (congestion_level * 0.3)  # Simplified inverse relationship
        else:
            average_speed = 0
        
        cap.release()
        
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'camera_id': camera_id,
            'vehicle_counts': vehicle_counts,
            'total_vehicles': total_vehicles,
            'congestion_level': congestion_level,
            'average_speed': average_speed
        }
    
    def detect_incidents(self, current_frame, previous_frame, camera_id):
        """Detect traffic incidents like accidents or stalled vehicles"""
        # This would use a combination of:
        # 1. Sudden changes in traffic flow
        # 2. Abnormal vehicle positioning
        # 3. Vehicle movement patterns
        
        # Simplified placeholder implementation
        # In a real system, this would use more sophisticated CV and anomaly detection
        
        # Convert frames to grayscale for processing
        if current_frame is None or previous_frame is None:
            return {'incident_detected': False}
            
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference to detect sudden changes
        frame_diff = cv2.absdiff(gray_current, gray_previous)
        _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(threshold, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for large stationary objects where there should be movement
        incident_detected = False
        incident_location = None
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum size threshold
                x, y, w, h = cv2.boundingRect(contour)
                # Check if object is in road area (would be defined per camera)
                # and if it's stationary for multiple frames
                incident_detected = True
                incident_location = (x + w//2, y + h//2)
                break
        
        if incident_detected:
            # Log the incident
            incident_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'camera_id': camera_id,
                'location': incident_location,
                'type': 'potential_accident',  # Would be classified more specifically in a real system
                'confidence': 0.75  # Placeholder
            }
            incident_collection.insert_one(incident_data)
            
            # In a real system, this would trigger alerts and response protocols
            return {
                'incident_detected': True,
                'location': incident_location,
                'confidence': 0.75
            }
        
        return {'incident_detected': False}

# 2. Traffic Prediction Model using Ensemble of LSTM and CNN
class TrafficPredictionModel:
    def __init__(self, config):
        self.window_size = config['model_params']['window_size']
        self.prediction_horizon = config['model_params']['prediction_horizon']
        self.batch_size = config['model_params']['batch_size']
        self.epochs = config['model_params']['epochs']
        self.model = self.build_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def build_model(self):
        """Build a hybrid model combining LSTM for temporal and CNN for spatial features"""
        # Temporal input branch (traffic flow time series)
        temporal_input = Input(shape=(self.window_size, 5))  # 5 features per time step
        lstm1 = LSTM(64, return_sequences=True)(temporal_input)
        lstm2 = LSTM(32)(lstm1)
        
        # Spatial input branch (road network features)
        spatial_input = Input(shape=(32, 32, 1))  # Grid representation of the area
        conv1 = Conv2D(32, (3, 3), activation='relu')(spatial_input)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        flatten = Flatten()(pool2)
        
        # Weather and event input branch
        context_input = Input(shape=(10,))  # Weather, events, time features
        
        # Merge all features
        merged = Concatenate()([lstm2, flatten, context_input])
        dense1 = Dense(128, activation='relu')(merged)
        dropout = Dropout(0.2)(dense1)
        dense2 = Dense(64, activation='relu')(dropout)
        output = Dense(self.prediction_horizon)(dense2)  # Predict congestion levels
        
        model = Model(inputs=[temporal_input, spatial_input, context_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def prepare_data(self, raw_data):
        """Process raw traffic data for model input"""
        # Extract features
        traffic_flow = []
        spatial_features = []
        context_features = []
        targets = []
        
        # Group data by location and time
        # This is simplified - actual implementation would handle complex spatiotemporal processing
        for location, location_data in raw_data.groupby('location_id'):
            location_data = location_data.sort_values('timestamp')
            
            # Extract flow data (vehicles per min, congestion level, etc.)
            flow_features = location_data[['vehicle_count', 'congestion_level', 
                                           'average_speed', 'heavy_vehicle_ratio', 
                                           'occupancy_rate']].values
            
            # Scale features
            flow_features_scaled = self.scaler.fit_transform(flow_features)
            
            # Create sequences for LSTM
            for i in range(len(flow_features_scaled) - self.window_size - self.prediction_horizon):
                traffic_flow.append(flow_features_scaled[i:i+self.window_size])
                
# Target values (future congestion levels)
                targets.append(flow_features_scaled[i+self.window_size:i+self.window_size+self.prediction_horizon, 1])
                
                # Spatial features (road network around this location)
                # In a real system, this would be a grid representation of the area
                spatial_grid = np.zeros((32, 32))
                # Populate grid with road network, intersections, etc.
                # This is placeholder code - actual implementation would use GIS data
                spatial_features.append(spatial_grid.reshape(32, 32, 1))
                
                # Context features (weather, events, time of day, day of week)
                time_features = self.extract_time_features(location_data.iloc[i+self.window_size]['timestamp'])
                weather = location_data.iloc[i+self.window_size][['temperature', 'precipitation', 'visibility']].values
                events = [0, 0]  # Binary flags for special events (holidays, sports events, etc.)
                context = np.concatenate([time_features, weather, events])
                context_features.append(context)
        
        return [
            np.array(traffic_flow),
            np.array(spatial_features),
            np.array(context_features)
        ], np.array(targets)
    
    def extract_time_features(self, timestamp):
        """Extract cyclical time features"""
        dt = pd.to_datetime(timestamp)
        
        # Hour of day (0-23) -> sin/cos encoding for cyclical nature
        hour = dt.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (0-6) -> sin/cos encoding
        dow = dt.dayofweek
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        
        return np.array([hour_sin, hour_cos, dow_sin, dow_cos, dt.month / 12])
    
    def train(self, raw_data):
        """Train the prediction model with historical data"""
        X, y = self.prepare_data(raw_data)
        self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint('models/traffic_prediction.h5', save_best_only=True)
            ]
        )
        
    def predict(self, current_data):
        """Generate traffic predictions"""
        X, _ = self.prepare_data(current_data)
        predictions = self.model.predict(X)
        
        # Inverse transform to get actual congestion levels
        # This is simplified - actual implementation would handle the full inverse transformation
        predictions_original_scale = self.scaler.inverse_transform(
            np.hstack([np.zeros((predictions.shape[0], 1)), predictions, np.zeros((predictions.shape[0], 3))])
        )[:, 1:self.prediction_horizon+1]
        
        return predictions_original_scale

# 3. Traffic Signal Optimization using Reinforcement Learning
class AdaptiveSignalController:
    def __init__(self, config):
        self.intersections = config['intersections']
        self.rl_models = {}
        self.initialize_rl_models()
        
    def initialize_rl_models(self):
        """Initialize RL models for each intersection"""
        for intersection_id, intersection_data in self.intersections.items():
            # Each intersection gets its own model
            # In a production system, we might use transfer learning between similar intersections
            
            # Define the state and action spaces
            num_phases = len(intersection_data['phases'])
            state_dim = 15  # Traffic counts, queues, waiting times, etc.
            action_dim = num_phases  # Different signal phases
            
            # Create a neural network for Q-learning
            model = Sequential([
                Dense(128, activation='relu', input_shape=(state_dim,)),
                Dense(64, activation='relu'),
                Dense(action_dim, activation='linear')
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            self.rl_models[intersection_id] = {
                'model': model,
                'state_dim': state_dim,
                'action_dim': action_dim
            }
    
    def get_state(self, intersection_id, traffic_data):
        """Convert current traffic conditions to state representation"""
        # This function would extract relevant features:
        # - Queue lengths for each approach
        # - Waiting times
        # - Incoming traffic volumes
        # - Current phase and time elapsed
        
        # Simplified placeholder implementation
        state = np.zeros(self.rl_models[intersection_id]['state_dim'])
        
        # First 8 values: queue lengths for each approach (N, S, E, W) * (through, turn)
        # This would come from real-time sensors or camera detection
        state[0:8] = np.random.randint(0, 15, 8)  # Placeholder random values
        
        # Next 4 values: arrival rates 
        state[8:12] = np.random.randint(0, 10, 4)  # Placeholder random values
        
        # Current phase, time elapsed, special conditions
        state[12] = 0  # Current phase index
        state[13] = 10  # Seconds elapsed in current phase
        state[14] = 0  # Special condition flag (0=normal, 1=emergency vehicle, etc.)
        
        return state
    
    def select_action(self, intersection_id, state, epsilon=0.1):
        """Choose the next signal phase using epsilon-greedy policy"""
        if np.random.random() < epsilon:
            # Exploration: random action
            return np.random.randint(0, self.rl_models[intersection_id]['action_dim'])
        else:
            # Exploitation: best known action
            q_values = self.rl_models[intersection_id]['model'].predict(state.reshape(1, -1))[0]
            return np.argmax(q_values)
    
    def update_model(self, intersection_id, state, action, reward, next_state, done):
        """Update the RL model using experience replay"""
        # In a real system, experiences would be stored in a replay buffer
        # and batch updates would be performed
        
        # Q-learning update
        q_values = self.rl_models[intersection_id]['model'].predict(state.reshape(1, -1))[0]
        next_q_values = self.rl_models[intersection_id]['model'].predict(next_state.reshape(1, -1))[0]
        
        # Bellman equation
        gamma = 0.95  # Discount factor
        q_values[action] = reward + gamma * np.max(next_q_values) * (1 - done)
        
        # Update model
        self.rl_models[intersection_id]['model'].fit(
            state.reshape(1, -1), 
            q_values.reshape(1, -1),
            verbose=0
        )
    
    def optimize_signals(self, traffic_data):
        """Generate optimal signal timings based on current and predicted traffic"""
        signal_commands = {}
        
        for intersection_id in self.intersections:
            # Get current state
            state = self.get_state(intersection_id, traffic_data)
            
            # Select best action
            action = self.select_action(intersection_id, state, epsilon=0.05)
            
            # Convert action to signal timing command
            # In a real system, this would set specific timings for each phase
            signal_commands[intersection_id] = {
                'next_phase': self.intersections[intersection_id]['phases'][action],
                'phase_duration': 30,  # Fixed 30 seconds for simplicity
                'timestamp': datetime.datetime.now().isoformat()
            }
        
        return signal_commands
    
    def coordinate_corridors(self, signal_commands):
        """Implement green wave coordination for major corridors"""
        # Identify corridors (sequences of intersections)
        corridors = [
            ['intersection_1', 'intersection_2', 'intersection_3'],
            ['intersection_4', 'intersection_5', 'intersection_6'],
            # Additional corridors would be defined here
        ]
        
        for corridor in corridors:
            # Calculate progressive timing offsets based on distance and speed
            # This is a simplified example - real implementation would be more sophisticated
            offset = 0
            travel_time = 20  # seconds between intersections
            
            for i, intersection_id in enumerate(corridor):
                if i > 0:
                    # Add offset to create green wave
                    signal_commands[intersection_id]['offset'] = offset
                    offset += travel_time
        
        return signal_commands

# 4. API and Communication Layer
class TrafficManagementAPI:
    def __init__(self, config):
        self.app = Flask(__name__)
        self.register_endpoints()
        self.mqtt_client = mqtt.Client()
        self.configure_mqtt()
        
    def register_endpoints(self):
        @self.app.route('/api/traffic/current', methods=['GET'])
        def get_current_traffic():
            """Return current traffic conditions"""
            # Query the most recent data from MongoDB
            current_data = list(traffic_collection.find().sort('timestamp', -1).limit(100))
            
            # Convert ObjectId to string for JSON serialization
            for item in current_data:
                item['_id'] = str(item['_id'])
            
            return jsonify(current_data)
        
        @self.app.route('/api/traffic/predict', methods=['GET'])
        def get_traffic_predictions():
            """Return traffic predictions"""
            # Fetch predictions from the prediction model
            # This would normally use the actual prediction model
            
            # Placeholder data
            predictions = []
            for i in range(10):
                predictions.append({
                    'location_id': f'location_{i}',
                    'timestamp': (datetime.datetime.now() + datetime.timedelta(hours=1)).isoformat(),
                    'predicted_congestion': np.random.randint(20, 80),
                    'confidence': np.random.uniform(0.7, 0.95)
                })
            
            return jsonify(predictions)
        
        @self.app.route('/api/incidents/current', methods=['GET'])
        def get_current_incidents():
            """Return current traffic incidents"""
            # Query the most recent incidents from MongoDB
            current_incidents = list(incident_collection.find().sort('timestamp', -1).limit(10))
            
            # Convert ObjectId to string for JSON serialization
            for item in current_incidents:
                item['_id'] = str(item['_id'])
            
            return jsonify(current_incidents)
        
        @self.app.route('/api/signals/status', methods=['GET'])
        def get_signal_status():
            """Return current traffic signal status"""
            # Query signal status from database
            # Placeholder implementation
            signal_status = {}
            for intersection_id in config['intersections']:
                signal_status[intersection_id] = {
                    'current_phase': np.random.choice(['NS', 'EW', 'N_TURN', 'S_TURN']),
                    'time_elapsed': np.random.randint(0, 60),
                    'next_phase': np.random.choice(['NS', 'EW', 'N_TURN', 'S_TURN'])
                }
            
            return jsonify(signal_status)
    
    def configure_mqtt(self):
        """Configure MQTT for IoT communication"""
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        
        # Connect to MQTT broker
        self.mqtt_client.connect('localhost', 1883, 60)
        
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """Subscribe to relevant MQTT topics on connection"""
        client.subscribe('traffic/sensors/#')
        client.subscribe('traffic/incidents/#')
        client.subscribe('traffic/signals/#')
        
    def on_mqtt_message(self, client, userdata, msg):
        """Process incoming MQTT messages"""
        topic = msg.topic
        payload = json.loads(msg.payload.decode())
        
        if topic.startswith('traffic/sensors/'):
            # Store sensor data
            traffic_collection.insert_one(payload)
            
            # Publish to Kafka for stream processing
            kafka_producer.produce(
                config['kafka']['traffic_data_topic'],
                key=topic.split('/')[-1],
                value=json.dumps(payload)
            )
        
        elif topic.startswith('traffic/incidents/'):
            # Store incident data
            incident_collection.insert_one(payload)
            
            # Trigger alerts or response protocols
            self.handle_incident(payload)
    
    def handle_incident(self, incident_data):
        """Handle traffic incidents with appropriate responses"""
        severity = incident_data.get('severity', 'medium')
        
        if severity == 'high':
            # High severity incidents trigger emergency response
            # This would integrate with emergency services in a real implementation
            print(f"EMERGENCY ALERT: High severity incident at {incident_data['location']}")
            
            # Adjust nearby traffic signals
            self.mqtt_client.publish(
                f"traffic/signals/emergency/{incident_data['nearby_intersection']}",
                json.dumps({
                    'command': 'emergency_mode',
                    'duration': 300,  # 5 minutes
                    'approach': incident_data['affected_approach']
                })
            )
        
        # Log incident for reporting
        print(f"Incident logged: {incident_data['type']} at {incident_data['location']}")

# Main integration - how the components work together
def main():
    # Initialize the system components
    vision_system = TrafficVisionSystem(config['camera_endpoints'])
    prediction_model = TrafficPredictionModel(config)
    signal_controller = AdaptiveSignalController(config)
    api = TrafficManagementAPI(config)
    
    # Data processing loop
    while True:
        # 1. Collect current traffic data from cameras
        traffic_data = {}
        for i, camera_endpoint in enumerate(config['camera_endpoints']):
            camera_data = vision_system.process_camera_feed(i)
            if camera_data:
                traffic_data[f'camera_{i}'] = camera_data
                
                # Store in database
                traffic_collection.insert_one(camera_data)
        
        # 2. Generate traffic predictions
        predictions = prediction_model.predict(traffic_data)
        
        # 3. Optimize signal timings
        signal_commands = signal_controller.optimize_signals(traffic_data)
        
        # 4. Apply corridor coordination
        coordinated_commands = signal_controller.coordinate_corridors(signal_commands)
        
        # 5. Send commands to traffic signals
        for intersection_id, command in coordinated_commands.items():
            # In a real system, this would use MQTT or a similar protocol
            # to communicate with the physical traffic controllers
            kafka_producer.produce(
                config['kafka']['signal_control_topic'],
                key=intersection_id,
                value=json.dumps(command)
            )
        
        # 6. Wait for next iteration
        time.sleep(60)  # Update every minute
